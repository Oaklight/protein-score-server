import logging
import os
import time
from threading import Semaphore

import requests
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

# Import the ratelimit library
from ratelimit import limits, sleep_and_retry
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

from task import PredictTask

torch.backends.cuda.matmul.allow_tf32 = True

global_rate_limiter = Semaphore(5)


class ProtModel:
    def __init__(self, model_name, id, **kwargs):
        self.model = None
        self.model_name = model_name
        self.id = id

        self.logger = logging.getLogger(f"ProtModel_{self.id}")
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

        self.gpu_model = False
        self.last_called = time.time()
        self.priority = 0

        self.states = {
            "busy": False,
            "task_id": None,
            "task_type": None,
            "pdb_done": False,
            "prev_states": None,
        }

        # ########################################
        #               ESM3 Small
        # ########################################

        if self.model_name == "esm3":
            self.gpu_model = True
            self.priority = 1

            # Initialize ESM3 model
            self.device = torch.device(
                kwargs.get("device", "cpu"),
            )
            self.model = ESM3.from_pretrained(
                "esm3_sm_open_v1",
                device=self.device,
            )
            self.esm_num_steps = kwargs.get("esm_num_steps", 5)
            self.logger.debug("ESM3 model loaded")

        # ########################################
        #           ESMFold Meta API
        # ########################################

        elif self.model_name == "esmfold":
            self.priority = 2

            # Initialize ESMFold API client
            self.esmfold_api_url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
            self.logger.debug(f"{self.id}-th ESMFold API client loaded")

        # ########################################
        #           HuggingFace ESMFold
        # ########################################

        elif self.model_name == "huggingface_esmfold":
            self.gpu_model = True
            self.priority = 0

            # Initialize Hugging Face ESMFold model
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            self.model = EsmForProteinFolding.from_pretrained(
                "facebook/esmfold_v1",
                low_cpu_mem_usage=True,
            )
            # move to device
            self.device = torch.device(
                kwargs.get("device", "cpu"),
            )
            self.model.to(self.device)

            self.model.esm = self.model.esm.half()
            if torch.cuda.is_available():
                torch.backends.cudnn.allow_tf32 = True
            self.model.trunk.set_chunk_size(64)
            self.logger.debug("Hugging Face ESMFold model loaded")
        else:
            raise ValueError("Unsupported model name")

    def __lt__(self, other):
        if self.priority == other.priority:
            return self.last_called < other.last_called
        return self.priority < other.priority

    def predict_structure(self, task: PredictTask, pdb_path):
        temp_pdb_path = os.path.join(
            pdb_path, f"result_{self.model_name}_{task.id}.pdb"
        )
        self.logger.debug(f"Generating structure for {task.id}")

        self.last_called = time.time()
        tmp_states = self.states.copy()
        tmp_states["prev_states"] = None
        self.states = {
            "busy": True,
            "task_id": task.id,
            "task_type": task.type,
            "pdb_done": None,
            "prev_states": tmp_states,
        }

        # ########################################
        #               ESM3 Small
        # ########################################

        if self.model_name == "esm3":

            # Use the ESM3 model to generate structure
            self.logger.debug("creating protein")
            protein = ESMProtein(sequence=task.seq)
            protein = self.model.generate(
                protein,
                GenerationConfig(track="structure", num_steps=self.esm_num_steps),
            )
            self.logger.debug("writing protein to pdb")
            protein.to_pdb(temp_pdb_path)
            del protein

        # ########################################
        #           ESMFold Meta API
        # ########################################

        elif self.model_name == "esmfold":
            # Use the ESMFold API to generate structure
            self.logger.debug("sending request")

            response = self.rate_limited_request(task.seq)

            # self.logger.debug(response.status_code)
            # self.logger.debug(response.text)
            if response.status_code == 200:
                self.states["pdb_done"] = "waiting to write"
                # save response.text into a pdb file
                self.logger.debug("writing response to pdb")
                with open(temp_pdb_path, "w") as pdb_file:
                    pdb_file.write(response.text)
                self.states["pdb_done"] = "written!"
            else:
                temp_pdb_path = None
                self.logger.debug(
                    f"error! API error: {response.status_code} | details: {response.text}"
                )

        # ########################################
        #           HuggingFace ESMFold
        # ########################################

        elif self.model_name == "huggingface_esmfold":

            # Use the Hugging Face ESMFold model to generate structure
            self.logger.debug(f"[{task.id}] tokenizing input")
            tokenized_input = self.tokenizer(
                [task.seq], return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(self.device)

            self.logger.debug(f"[{task.id}] running inference")
            with torch.no_grad():
                outputs = self.model(tokenized_input)

            self.logger.debug(f"[{task.id}] converting outputs to pdb")

            pdb_content = convert_outputs_to_pdb(outputs)

            self.logger.debug(f"[{task.id}] writing pdb to temp file")
            with open(temp_pdb_path, "w") as pdb_file:
                pdb_file.write("\n".join(pdb_content))

        else:
            raise ValueError("Unsupported model name")

        self.states["pdb_done"] = True
        self.states["busy"] = False

        return temp_pdb_path

    @sleep_and_retry
    @limits(calls=5, period=1)  # 5 calls per second
    def rate_limited_request(self, seq):
        with global_rate_limiter:
            response = requests.post(self.esmfold_api_url, data=seq)
        return response


def convert_outputs_to_pdb(outputs):

    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = outputs
    outputs = {k: v.cpu().numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=(
                outputs["chain_index"][i] if "chain_index" in outputs else None
            ),
        )
        pdbs.append(to_pdb(pred))
    return pdbs
