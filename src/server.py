import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import gc
import json
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import time

import biotite.structure.io as bsio
import torch
import yaml
from huggingface_hub import login
from TMscore import TMscore

import to_pdb
from cache import ProtCache
from model import ProtModel
from task import PredictTask

torch.set_warn_always(False)
to_pdb.process()


class PredictServer:
    def __init__(self, config_file, logger):
        self.logger = logger

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        with open(self.config["backbone_pdb"]["reversed_index"]) as f:
            self.reversed_index = yaml.safe_load(f)

        login(self.config["api_key"])
        self.load_models()

        # initialize task queue, result pool, and result queue
        self.task_queue = queue.Queue(self.config["task_queue_size"])
        self.working_pool = set()
        self.result_pool = {}
        self.lock_workingpool = Lock()
        self.lock_resultpool = Lock()
        self.prepare_cache(load_history=True)

        # initialize thread main executor and process pool executor
        self.main_executor = threading.Thread(target=self.run, daemon=True)
        self.main_executor.start()

    def load_models(self):
        model_name, placements = (
            self.config["model"]["name"],
            self.config["model"]["replica"],
        )
        self.logger.info(placements)
        total_replica = sum(placements.values())
        self.model_avail = queue.Queue(total_replica)

        t_loading = time()
        for cuda_idx, replica_num in placements.items():
            self.logger.info(cuda_idx)
            for j in range(replica_num):

                if model_name == "esm3":
                    model = ProtModel(
                        model_name,
                        id=j,
                        device=f"cuda:{cuda_idx}",
                        esm_num_steps=self.config["model"]["esm_num_steps"],
                    )
                elif model_name == "esmfold":
                    model = ProtModel(model_name, id=j)
                else:
                    raise ValueError("Unsupported model name")

                self.model_avail.put(model)
                self.logger.info(f"cuda:{cuda_idx} [{j}]-th Model {model_name} loaded")
        t_loading_done = time() - t_loading

        self.model_executor = ThreadPoolExecutor(max_workers=total_replica)
        self.logger.info(f"{placements} Models loaded in {t_loading_done} seconds")

    def prepare_cache(self, load_history=False):
        if not os.path.exists(self.config["intermediate_pdb_path"]):
            os.makedirs(self.config["intermediate_pdb_path"], exist_ok=True)
            self.logger.info(f"Creating {self.config['intermediate_pdb_path']}")

        self.cache_db = ProtCache(self.config["cache_db_path"])

    def get_available_model(self):
        return self.model_avail.get()

    def release_model(self, model):
        self.model_avail.put(model)
        # Release GPU memory
        if model.model_name in ["esm3"]:
            torch.cuda.empty_cache()
            gc.collect()

    def _predict_core(self, task: PredictTask, model: ProtModel):
        t_predict = time()
        output_data = None

        # either hit pdb cache or compute new pdb
        temp_pdb_path = self.cache_db.get(task.seq, table="prot_cache")
        self.logger.debug(f"[{task.id}] cache get result: {temp_pdb_path}")
        self.logger.debug("strange!")
        if temp_pdb_path is None:
            temp_pdb_path = model.predict_structure(
                task, self.config["intermediate_pdb_path"]
            )
            self.logger.debug(
                f"[{task.id}] pdb predicted, result saved to {temp_pdb_path}"
            )
            self.cache_db.set(task.seq, temp_pdb_path, table="prot_cache")
        elif not os.path.exists(temp_pdb_path):
            temp_pdb_path = model.predict_structure(
                task, self.config["intermediate_pdb_path"]
            )
            self.logger.debug(
                f"[{task.id}] pdb predicted, result saved to {temp_pdb_path}"
            )
            self.cache_db.set(task.seq, temp_pdb_path, table="prot_cache")
        else:
            self.logger.debug(
                f"[{task.id}] structure prediction cached, result loaded from {temp_pdb_path}"
            )

        self.logger.info(
            f"[{task.id}] structure prediction done, result saved to {temp_pdb_path}"
        )

        try:
            match task.type:

                case "plddt":
                    self.logger.debug(f"[{task.id}] Task type is 'plddt'")

                    struct = bsio.load_structure(
                        temp_pdb_path, extra_fields=["b_factor"]
                    )
                    # this will be the pLDDT, convert to float
                    plddt = struct.b_factor.mean().item()
                    self.logger.debug(f"[{task.id}] pLDDT is {plddt}")

                    output_data = plddt

                case "tmscore":
                    self.logger.debug(f"[{task.id}] Task type is 'tmscore'")
                    reference_pdb = to_pdb.get_pdb_file(self.reversed_index, task.name)
                    self.logger.debug(f"[{task.id}] Reference pdb is {reference_pdb}")

                    lengths, results = TMscore(reference_pdb, temp_pdb_path)
                    self.logger.debug(f"[{task.id}] Alignment done")

                    self.logger.debug(f"[{task.id}] TMscore is {results}")

                    output_data = results[0]

                case _:
                    # not implement
                    raise Exception("Task type not supported")

        except Exception as e:
            self.logger.error(f"[{task.id}] Task failed: {e}")
            raise e

        # remove task from working pool
        t_predict_done = time() - t_predict
        self.logger.info(f"[{task.id}] Task done in {t_predict_done:.2f} seconds")

        return output_data

    def predict(self, task: PredictTask, model):
        with self.lock_workingpool:
            self.working_pool.add(task.id)
        self.logger.debug(f"[{task.id}] working_pool lock | add to working pool")
        self.logger.info(self.working_pool)

        output_data = self.cache_db.get(task.hash, table="task_cache")
        if output_data is not None:
            self.logger.debug(f"[{task.id}] Task hit cache: {output_data}")
        else:
            output_data = self._predict_core(task, model)
            self.logger.debug(f"[{task.id}] {output_data}")
        # add result to cache
        self.cache_db.set(task.hash, output_data, table="task_cache")

        # put model back to available queue
        self.release_model(model)

        # put future into result pool
        with self.lock_resultpool:
            self.result_pool[task.id] = output_data
        self.logger.debug(f"[{task.id}] result_pool lock | result put into result pool")
        with self.lock_workingpool:
            self.working_pool.remove(task.id)
        self.logger.debug(
            f"[{task.id}] work_pool lock | item removed from working pool"
        )

        self.logger.info(f"[{task.id}] Task done, result put into result pool")
        # self.logger.debug(self.result_pool)

    def run(self):
        self.logger.info("Server is running")
        while True:
            task = self.task_queue.get()
            if task is None:
                self.logger.info("processing thread received None, exiting...")
                break

            model = self.get_available_model()  # blocking until a model is available
            self.model_executor.submit(self.predict, task, model)
            self.logger.info(f"[{task.id}] Task submitted to pool executor")

    def stop_server(self):
        self.logger.info("Stopping server...")

        # signal the job queue thread to exit
        self.task_queue.put(None)
        # wait for the job queue thread to exit
        self.main_executor.join()
        # shutdown the executors
        self.model_executor.shutdown(wait=True)
        # shutdown the cache db
        self.cache_db.close()

        self.logger.info("Server stopped. Bye!")

    def get_status(self):
        with self.lock_workingpool:
            busy_models = len(self.working_pool)
        with self.lock_resultpool:
            processed_tasks = len(self.result_pool)
        remaining_tasks = self.task_queue.qsize()

        return {
            "busy_models": busy_models,
            "processed_tasks": processed_tasks,
            "remaining_tasks": remaining_tasks,
        }


if __name__ == "__main__":
    task = PredictTask(
        seq="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        name="1a1x.A",
        task_type="plddt",
    )

    print(json.dumps(task.to_dict(), indent=4))
