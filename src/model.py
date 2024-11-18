import logging
from time import time

import requests
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

from task import PredictTask


class ProtModel:
    def __init__(self, model_name, id, **kwargs):
        self.model = None
        self.model_name = model_name
        self.id = id

        self.logger = logging.getLogger(f"ProtModel_{self.id}")
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.DEBUG)

        if self.model_name == "esm3":
            # Initialize ESM3 model
            self.model = ESM3.from_pretrained(
                "esm3_sm_open_v1",
                device=torch.device(
                    kwargs.get("device", "cpu"),
                ),
            )
            self.esm_num_steps = kwargs.get("esm_num_steps", 5)
            self.logger.debug("ESM3 model loaded")
        elif self.model_name == "esmfold":
            # Initialize ESMFold API client
            self.esmfold_api_url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
            self.logger.debug(f"{self.id}-th ESMFold API client loaded")
        else:
            raise ValueError("Unsupported model name")

    def predict_structure(self, task: PredictTask, pdb_path):
        temp_pdb_path = os.path.join(
            pdb_path, f"result_{self.model_name}_{task.id}.pdb"
        )
        self.logger.debug(f"Generating structure for {task.id}")

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

        elif self.model_name == "esmfold":
            # Use the ESMFold API to generate structure
            self.logger.debug("sending request")

            # update LAST_ESMFOLD_REQUEST_TIME
            global LAST_ESMFOLD_REQUEST_TIME
            if time() - LAST_ESMFOLD_REQUEST_TIME < 0.5:
                time.sleep(0.5)
            LAST_ESMFOLD_REQUEST_TIME = time()

            response = requests.post(self.esmfold_api_url, data=task.seq)
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
        else:
            raise ValueError("Unsupported model name")

        return temp_pdb_path
