import logging
import os
from time import time

import requests
from transformers import AutoTokenizer, EsmForProteinFolding

from task import PredictTask

LAST_ESMFOLD_REQUEST_TIME = 0

import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

torch.backends.cuda.matmul.allow_tf32 = True


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

        # ########################################
        #               ESM3 Small
        # ########################################

        if self.model_name == "esm3":

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
            # Initialize ESMFold API client
            self.esmfold_api_url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
            self.logger.debug(f"{self.id}-th ESMFold API client loaded")

        # ########################################
        #           HuggingFace ESMFold
        # ########################################

        elif self.model_name == "huggingface_esmfold":

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

    def predict_structure(self, task: PredictTask, pdb_path):
        temp_pdb_path = os.path.join(
            pdb_path, f"result_{self.model_name}_{task.id}.pdb"
        )
        self.logger.debug(f"Generating structure for {task.id}")

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

            # update LAST_ESMFOLD_REQUEST_TIME
            global LAST_ESMFOLD_REQUEST_TIME
            if time() - LAST_ESMFOLD_REQUEST_TIME < 0.5:
                time.sleep(0.5)
            LAST_ESMFOLD_REQUEST_TIME = time()

            response = requests.post(self.esmfold_api_url, data=task.seq)
            self.logger.debug(response.status_code)
            self.logger.debug(response.text)
            # self.logger.debug(f"{response.text}")
            if response.status_code == 200:
                # save response.text into a pdb file
                self.logger.debug("writing response to pdb")
                with open(temp_pdb_path, "w") as pdb_file:
                    pdb_file.write(response.text)
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

        return temp_pdb_path


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
