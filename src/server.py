import os
import sys
from turtle import color

from utils import colorstring

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import heapq
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock

import biotite.structure.io as bsio
import torch
import yaml
from huggingface_hub import login
from TMscore import TMscore

import to_pdb
from cache import ProtCache
from model import ProtModel
from scheduler import TASK_STATES, TaskScheduler
from task import PredictTask

torch.set_warn_always(False)


class PredictServer:
    def __init__(self, config_file, logger):
        self.logger = logger

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        with open(self.config["backbone_pdb"]["reversed_index"]) as f:
            self.reversed_index = yaml.safe_load(f)

        with open(self.config["backbone_pdb"]["name_seq_map"]) as f:
            self.name_seq_map = yaml.safe_load(f)

        login(self.config["api_key"])
        self.load_models()

        # initialize task scheduler
        self.task_scheduler = TaskScheduler()
        self.prepare_cache(load_history=True)

        # initialize thread main executor and process pool executor
        self.main_executor = threading.Thread(target=self.run, daemon=True)
        self.main_executor.start()

    def _add_model(self, model, is_init=False):
        """
        Adds a model to the available models list and handles GPU memory management.

        Parameters:
        model (ProtModel): The model to be added.
        is_init (bool): Indicates if this is the initial model load.

        The function appends the model to the model_avail list and performs a heapify
        operation if necessary. It also manages GPU memory by clearing the cache if the
        GPU memory usage exceeds 80% of the total memory or if it's an initial model load.
        """
        with self.lock_model:
            self.model_avail.append(model)  # defer heapify to get_avail_model

        if model.gpu_model:
            # only clear cache when needed
            if is_init or (
                torch.cuda.memory_allocated(model.device)
                > 0.8 * torch.cuda.get_device_properties(model.device).total_memory
            ):
                torch.cuda.empty_cache()

    def load_models(self):
        model_name, placements = (
            self.config["model"]["name"],
            self.config["model"]["replica"],
        )
        self.logger.info(placements)
        total_replica = sum(placements.values())
        self.model_avail = []  # use heaqq to maintain
        self.model_status = []
        self.lock_model = Lock()

        t_loading = time.time()
        all_idx = -1
        for cuda_idx, replica_num in placements.items():
            self.logger.info(cuda_idx)
            for j in range(replica_num):
                all_idx += 1
                # if model_name == "esm3":
                #     model = ProtModel(
                #         model_name,
                #         id=all_idx,
                #         device=f"cuda:{cuda_idx}",
                #         esm_num_steps=self.config["model"]["esm_num_steps"],
                #     )
                if model_name == "esmfold":
                    model = ProtModel(model_name, id=all_idx)
                elif model_name == "huggingface_esmfold":
                    model = ProtModel(
                        model_name,
                        id=all_idx,
                        device=f"cuda:{cuda_idx}",
                    )
                elif model_name == "esmfold_hybrid":
                    if cuda_idx == "_":  # api part
                        model = ProtModel("esmfold", id=all_idx)
                    else:
                        model = ProtModel(
                            "huggingface_esmfold",
                            id=all_idx,
                            device=f"cuda:{cuda_idx}",
                        )
                elif model_name in ["alphafold3", "protenix"]:
                    model = ProtModel(
                        model_name,
                        id=all_idx,
                        device=f"cuda:{cuda_idx}",
                    )
                else:
                    raise ValueError("Unsupported model name")

                self._add_model(model, is_init=True)
                self.model_status.append(model)

                self.logger.info(f"cuda:{cuda_idx} [{j}]-th Model {model_name} loaded")

        t_loading_done = time.time() - t_loading

        self.model_executor = ThreadPoolExecutor(max_workers=total_replica)
        self.logger.info(f"{placements} Models loaded in {t_loading_done} seconds")

    def prepare_cache(self, load_history=False):
        if not os.path.exists(self.config["intermediate_pdb_path"]):
            os.makedirs(self.config["intermediate_pdb_path"], exist_ok=True)
            self.logger.info(f"Creating {self.config['intermediate_pdb_path']}")

        self.cache_db = ProtCache(self.config["cache_db_path"])

    def get_avail_model(self, require_gpu=False):
        """
        Retrieves an available model from the model queue.

        Args:
            require_gpu (bool): Whether the model needs a GPU.

        Returns:
            ProtModel or None: An available model if found, else None.
        """
        model = None
        failed_get_model = 0
        max_fail = 5

        while model is None and failed_get_model < max_fail:
            with self.lock_model:
                try:
                    if require_gpu:
                        # Heapify the entire list to maintain the heap property
                        heapq.heapify(self.model_avail)
                        model = heapq.heappop(self.model_avail)
                    else:
                        # Randomly select an element and remove it
                        if self.model_avail:
                            model = random.choice(self.model_avail)
                            self.model_avail.remove(model)
                except IndexError:
                    model = None

            if model is None:  # sleep after releasing the lock
                # exponential backoff
                time.sleep(2**failed_get_model)
                failed_get_model += 1
        return model

    def release_model(self, model):
        self._add_model(model)

    def _predict_structure(
        self, task: PredictTask, model: ProtModel, for_seq2: bool = False
    ) -> str:
        """
        Predicts the protein structure for a given task. If a cached structure exists,
        it loads the cached structure; otherwise, it computes a new structure and caches it.

        Args:
            task (PredictTask): The task containing the sequence information for structure prediction.
            model (ProtModel): The model used for predicting the protein structure.
            for_seq2 (bool, optional): If True, uses the secondary sequence (seq2) from the task
                                    for prediction; otherwise, uses the primary sequence (seq).
                                    Defaults to False.

        Returns:
            str: The file path to the predicted or cached protein structure PDB file.
        """
        # either hit pdb cache or compute new pdb
        if for_seq2:
            temp_pdb_path = self.cache_db.get(task.seq2, table="prot_cache")
        else:
            temp_pdb_path = self.cache_db.get(task.seq, table="prot_cache")
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
                f"[{task.id}] pdb cached, result loaded from {temp_pdb_path}"
            )

        self.logger.info(
            f"[{task.id}] structure prediction done, result at {temp_pdb_path}"
        )
        return temp_pdb_path

    def _downstream_task(
        self, task: PredictTask, pdb_path, pdb_path2=None
    ) -> float | str:
        """
        Executes a downstream task based on the task type specified in the PredictTask object.

        Parameters:
        ----------
        task : PredictTask
            An object representing the task to be executed. It includes the task type and name.
        pdb_path : str
            The file path to the PDB file needed for the task.
        pdb_path2 : str, optional
            The file path to the second PDB file, required for tasks that involve comparing two PDB structures.

        Returns:
        -------
        float | str
            - If the task type is "pdb", returns the content of the PDB file as a string.
            - If the task type is "plddt", returns the mean pLDDT value as a float.
            - If the task type is "tmscore" or "sc-tmscore", returns the TMscore value as a float.

        Raises:
        -------
        Exception
            If the task type is not supported, an exception is logged, and None is returned.
        """
        output_data = None
        match task.type:
            case "pdb":
                self.logger.debug(f"[{task.id}] Task type is 'pdb'")
                # read the content of the pdb and return it
                with open(pdb_path, "r") as f:
                    output_data = f.read()
                self.logger.debug(f"[{task.id}] PDB content read and returned")

            case "plddt":
                self.logger.debug(f"[{task.id}] Task type is 'plddt'")

                struct = bsio.load_structure(pdb_path, extra_fields=["b_factor"])
                # this will be the pLDDT, convert to float
                plddt = struct.b_factor.mean().item()
                self.logger.debug(f"[{task.id}] pLDDT is {plddt}")

                output_data = plddt

            case "tmscore":
                self.logger.debug(f"[{task.id}] Task type is 'tmscore'")
                reference_pdb_path = to_pdb.get_pdb_file(self.reversed_index, task.name)
                self.logger.debug(f"[{task.id}] Reference pdb is {reference_pdb_path}")

                lengths, results = TMscore(reference_pdb_path, pdb_path)
                self.logger.debug(f"[{task.id}] Alignment done")

                self.logger.debug(f"[{task.id}] TMscore is {results}")

                output_data = results[0]

            case "sc-tmscore":
                self.logger.debug(f"[{task.id}] Task type is 'sc-tmscore'")
                reference_pdb_path = pdb_path2
                self.logger.debug(f"[{task.id}] Reference pdb is {reference_pdb_path}")

                lengths, results = TMscore(reference_pdb_path, pdb_path)
                self.logger.debug(f"[{task.id}] Alignment done")

                self.logger.debug(f"[{task.id}] sc-TMscore is {results}")

                output_data = results[0]

            case _:
                # not implement
                self.logger.error(Exception("Task type not supported"))

        return output_data

    def _predict_core(self, task: PredictTask, model: ProtModel):
        # ========== predict structure ==========
        temp_pdb_path = None
        try:
            temp_pdb_path2 = None
            if task.seq2:
                temp_pdb_path2 = self._predict_structure(task, model, for_seq2=True)
            temp_pdb_path = self._predict_structure(task, model)
        except Exception as e:
            self.logger.error(f"[{task.id}] Task failed: {e}")
            # raise custom error: structure_failure, you need to define it
        if temp_pdb_path is None:
            raise Exception("structure_failure")

        # ========== predict score ==========
        output_data = None
        try:
            output_data = self._downstream_task(task, temp_pdb_path, temp_pdb_path2)
        except Exception as e:
            self.logger.error(f"[{task.id}] Task failed: {e}")
        if output_data is None:
            raise Exception("score_failure")

        return output_data

    def predict(self, task: PredictTask):
        model = self.get_avail_model(task.require_gpu)
        if model is None:
            self.logger.warning(f"[{task.id}] No model available!")
            self.task_scheduler.reschedule_task(task.id, datetime.now())
            return

        # Add task to scheduler
        self.task_scheduler.change_task_status(task.id, "PROCESSING")

        # preprocess task, find seq for seq2 if only name provided
        if task.name:
            temp_seq2 = to_pdb.get_sequence_by_name_from_pdbbank(task.name)
            if temp_seq2:
                task.seq2 = temp_seq2
            else:
                self.logger.warning(
                    f"[{task.id}] No sequence found for name {task.name}"
                )
                if task.type == "sc-tmscore":
                    task.type = "tmscore"
                    self.logger.warning(
                        f"[{task.id}] Task type changed to 'tmscore' due to missing sequence for name {task.name}"
                    )

        try:
            output_data = self.cache_db.get(task.hash, table="task_cache")
            if output_data is not None:
                self.logger.debug(f"[{task.id}] Task hit cache: {output_data}")
            else:
                output_data = self._predict_core(task, model)
                self.logger.debug(f"[{task.id}] {output_data}")
            # add result to cache
            self.cache_db.set(task.hash, output_data, table="task_cache")

            # Update task status. If this fail (unlikely), exception could still be handled. It ensures atomicity and "happy path first" principle.
            self.task_scheduler.task_db[task.id]["result"] = output_data
            self.task_scheduler.change_task_status(task.id, "COMPLETED")

        except Exception as e:
            self.logger.error(f"[{task.id}] Task failed: {e}")
            # Update task status and reschedule
            self.task_scheduler.change_task_status(task.id, "FAILED")
            self.task_scheduler.reschedule_task(
                task.id, datetime.now(), need_expedite=True
            )
        finally:
            # put model back to available queue
            self.release_model(model)

    def run(self):
        self.logger.info("Server is running")
        while True:
            task_id = self.task_scheduler.get_next_task_with_max_priority()
            if task_id is None:
                self.logger.warning(
                    colorstring("No task available, waiting...", "yellow")
                )
                time.sleep(0.5)
                continue

            self.logger.info(colorstring(f"[{task_id}] Task picked up", "magenta"))
            if task_id == "shutdown":
                self.logger.warning("processing thread received `shutdown`, exiting...")
                break
            task = self.task_scheduler.get_task(task_id)

            self.model_executor.submit(self.predict, task)
            self.logger.info(f"[{task.id}] Task submitted to pool executor")

    def stop_server(self):
        self.logger.info("Stopping server...")

        count_down = 30  # 30 seconds before force stop

        # signal the job queue thread to exit
        self.task_scheduler.add_task(None)
        # wait for the job queue thread to exit
        self.main_executor.join(timeout=count_down)
        # shutdown the executors
        self.model_executor.shutdown(wait=True)
        # shutdown the cache db
        self.cache_db.close()

        self.logger.info("Server stopped. Bye!")

    def get_status(self):
        with self.lock_model:
            # busy_models = self.model_avail.maxsize - self.model_avail.qsize()
            busy_models_details = {
                f"[{each.id}]_{each.model_name}": each.states["busy"]
                for each in self.model_status
            }
            busy_models = sum(list(busy_models_details.values()))

            # find out which models are busy
            model_status = [
                {
                    "model_name": each.model_name,
                    "model_id": each.id,
                    "model_states": each.states.copy(),
                }
                for each in self.model_status
                if each.states["busy"]
            ]

        # Get task status from scheduler
        pending_tasks = len(
            [
                t
                for t in self.task_scheduler.task_db.values()
                if t["status"] == TASK_STATES["PENDING"]
            ]
        )
        completed_tasks = len(
            [
                t
                for t in self.task_scheduler.task_db.values()
                if t["status"] == TASK_STATES["COMPLETED"]
            ]
        )
        processing_tasks = [
            t
            for t in self.task_scheduler.task_db.values()
            if t["status"] == TASK_STATES["PROCESSING"]
        ]

        self.logger.info(json.dumps(model_status, indent=4))

        return {
            "busy_models": busy_models,
            "busy_models_details": busy_models_details,
            "processed_tasks": completed_tasks,
            "remaining_tasks": pending_tasks,
            "working_tasks": processing_tasks,
        }


if __name__ == "__main__":
    task = PredictTask(
        seq="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        name="1a1x.A",
        task_type="plddt",
    )

    print(json.dumps(task.to_dict(), indent=4))
