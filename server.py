import gc
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from time import time
from uuid import uuid4

import biotite.structure.io as bsio
import torch
import yaml
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from huggingface_hub import login

torch.set_warn_always(True)


class PredictTask:
    def __init__(self, id, seq, name, type):
        self.id = uuid4() if id is None else id
        self.seq = seq
        self.name = name
        self.type = type


class PredictServer:
    def __init__(self, config_file, logger):
        self.logger = logger

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        login(self.config["api_key"])
        self.load_models()
        self.esm_num_steps = self.config["model"]["esm_num_steps"]

        # initialize task queue, result pool, and result queue
        self.task_queue = queue.Queue(self.config["task_queue_size"])
        self.working_pool = set()
        self.result_pool = {}
        self.load_history_results()

        # initialize thread main executor and process pool executor
        self.main_executor = threading.Thread(target=self.run, daemon=True)
        self.main_executor.start()

    def load_models(self):
        model_name, replica = (
            self.config["model"]["name"],
            self.config["model"]["replica"],
        )

        placements = self.config["model"]["replica"]
        total_replica = sum(placements.values())
        self.model_avail = queue.Queue(total_replica)

        t_loading = time()
        for cuda_idx, replica_num in placements.items():
            for j in range(replica_num):
                model: ESM3InferenceClient = ESM3.from_pretrained(
                    "esm3_sm_open_v1"
                ).cuda(cuda_idx)
                self.model_avail.put(model)
                self.logger.info(f"cuda:{cuda_idx} [{j}]-th Model {model_name} loaded")
        t_loading_done = time() - t_loading

        self.model_executor = ThreadPoolExecutor(max_workers=total_replica)
        self.logger.info(f"{replica} Models loaded in {t_loading_done} seconds")

    def load_history_results(self):
        if not os.path.exists(self.config["intermediate_pdb_path"]):
            os.makedirs(self.config["intermediate_pdb_path"], exist_ok=True)
            self.logger.info(f"Creating {self.config['intermediate_pdb_path']}")
        if not os.path.exists(self.config["history_path"]):
            return
        with open(self.config["history_path"], "r") as f:
            self.logger.info(f"Loading {self.config['history_path']}...")
            bulk_result = f.read()
            for line in bulk_result.strip().split("\n"):
                # skip if line is empty
                if not line.strip():
                    continue
                task_id, result = [each.strip() for each in line.strip().split("\t")]
                self.result_pool[task_id] = float(result)

    def get_available_model(self):
        return self.model_avail.get()

    def release_model(self, model):
        self.model_avail.put(model)

    def predict(self, task: PredictTask, model):
        self.working_pool.add(task.id)

        t_predict = time()
        output_data = None

        protein, temp_pdb_path = self.predict_structure(task, model)
        self.logger.debug(f"[{task.id}] Task done, result saved to {temp_pdb_path}")

        match task.type:

            case "plddt":
                self.logger.debug(f"[{task.id}] Task type is 'plddt'")

                struct = bsio.load_structure(temp_pdb_path, extra_fields=["b_factor"])
                # this will be the pLDDT, convert to float
                output_data = struct.b_factor.mean().item()
                self.logger.debug(f"[{task.id}] pLDDT is {output_data}")

            case "tmscore":
                self.logger.debug(f"[{task.id}] Task type is 'tmscore'")

                # alignment = tmscoring.TMscoring("structure1.pdb", "structure2.pdb")

            case _:
                # not implement
                raise Exception("Task type not supported")

        t_predict_done = time() - t_predict
        self.release_model(model)  # put model back to available queue

        # put future into result pool
        self.result_pool[task.id] = output_data
        self.working_pool.remove(task.id)

        self.logger.debug(f"[{task.id}] Task done, result put into result pool")

        # Release GPU memory
        del protein
        del struct
        del output_data
        torch.cuda.empty_cache()
        gc.collect()

        self.logger.debug(self.result_pool)
        return output_data

    def predict_structure(self, task, model):
        protein = ESMProtein(sequence=task.seq)
        protein = model.generate(
            protein,
            GenerationConfig(track="structure", num_steps=self.esm_num_steps),
        )
        temp_pdb_path = os.path.join(
            self.config["intermediate_pdb_path"], f"result_esm3_{task.id}.pdb"
        )
        protein.to_pdb(temp_pdb_path)
        return protein, temp_pdb_path

    def run(self):
        self.logger.info("Server is running")
        while True:
            task = self.task_queue.get()
            if task is None:
                self.logger.debug("processing thread received None, exiting...")
                break
            self.logger.debug(f"[{task.id}] Task received")

            model = self.get_available_model()  # blocking until a model is available
            self.model_executor.submit(self.predict, task, model)
            self.logger.debug(f"[{task.id}] Task submitted to pool executor")

    def stop_server(self):
        self.logger.info("Stopping server...")

        # signal the job queue thread to exit
        self.task_queue.put(None)
        # wait for the job queue thread to exit
        self.main_executor.join()
        # shutdown the executors
        self.model_executor.shutdown(wait=True)

        # wait for the history dump thread to exit
        self.stop = True

        self.logger.info("Server stopped. Bye!")
