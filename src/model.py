import json
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import time
from tempfile import TemporaryDirectory
from threading import Semaphore
from typing import Any, Optional

import numpy as np
import requests
import torch
from biotite.structure.io import pdb, pdbx
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig
from protenix.config import parse_configs

# Import the ratelimit library
from ratelimit import limits, sleep_and_retry

# Import for protenix
from runner.batch_inference import get_default_runner

# Import for protenix
from runner.inference import InferenceRunner, download_infercence_cache, infer_predict
from runner.msa_search import contain_msa_res, msa_search_update

# Import for Huggingface ESM
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

            # Initialize HuggingFace ESMFold model
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

        # ########################################
        #           Protenix InferenceRunner
        # ########################################

        elif self.model_name in ["alphafold3", "protenix"]:
            self.gpu_model = True
            self.priority = 0

            # Initialize Protenix InferenceRunner
            self.seeds = [kwargs.get("seeds", 101)]

            # Set up temporary directories for input and output
            self.temp_outdir = "./temp_protenix"
            self.input_json_temp_dir = os.path.join(self.temp_outdir, "input_json")
            self.output_temp_dir = os.path.join(self.temp_outdir, "output")
            os.makedirs(self.input_json_temp_dir, exist_ok=True)
            os.makedirs(self.output_temp_dir, exist_ok=True)

            # Set the desired device for this model
            self.device_id = kwargs.get("device", "cpu")  # Default to CPU
            self.model = get_default_runner(
                seeds=self.seeds,
                device=self.device_id,  # cpu or cuda:{idx}
                dump_dir=self.output_temp_dir,
            )

            self.logger.debug(
                f"Protenix InferenceRunner loaded on {'CPU' if self.device_id == 'cpu' else f'GPU {self.device_id}'}"
            )

            self.logger.debug(f"AlphaFold3/Protenix InferenceRunner loaded")

        else:
            raise ValueError("Unsupported model name")

    def __lt__(self, other):
        """
        Compare two ProtModel instances based on priority and last called time.

        Args:
            other (ProtModel): Another ProtModel instance to compare against.

        Returns:
            bool: True if this model has a higher priority or the same priority and
                  was last called before the other model, False otherwise.
        """
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

        # ########################################
        #           Protenix InferenceRunner
        # ########################################

        elif self.model_name in ["alphafold3", "protenix"]:

            # Convert task.seq to the required JSON format
            self.logger.debug(
                f"[{task.id}] converting input to Protenix specific json input"
            )
            # Convert task.seq to the required JSON format
            infer_json = os.path.join(self.input_json_temp_dir, f"input_{task.id}.json")
            # save json to temp_dir
            with open(infer_json, "w") as f:
                json.dump(convert_seq_to_json(task), f)

            infer_json = msa_search_update(infer_json, self.input_json_temp_dir)

            self.logger.debug(
                f"[{task.id}] MSA search done, {json.dumps(infer_json,indent=2)}"
            )
            configs = self.model.configs
            configs["input_json_path"] = infer_json
            if not contain_msa_res(infer_json):
                raise RuntimeError(
                    f"`{infer_json}` has no msa result for `proteinChain`, please add first."
                )

            self.logger.debug(f"[{task.id}] running inference")
            infer_predict(self.model, configs)

            # result dump to pdb
            protenix_cif_path = os.path.join(
                self.output_temp_dir,
                task.id,
                f"seed_{configs['seeds'][0]}",
                "predictions",
                f"{task.id}_seed_{configs['seeds'][0]}_sample_0.cif",
            )
            protenix_pdb_path = temp_pdb_path

            self.logger.debug(f"[{task.id}] writing pdb to temp file")
            cif_to_pdb(protenix_cif_path, protenix_pdb_path)

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


class SingleGPUInferenceRunner(InferenceRunner):
    def __init__(self, configs: Any) -> None:
        super().__init__(configs)  # Call the superclass's __init__ method
        print(self.configs.dump_dir)

    def init_env(
        self,
    ) -> None:

        config_device = self.configs.get("device", "cpu")

        self.use_cuda = torch.cuda.is_available() and not config_device == "cpu"
        if self.use_cuda:
            self.device = torch.device(config_device)
            logging.info(f"Using CUDA_VISIBLE_DEVICES: [{self.device}]")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set `CUTLASS_PATH` env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            if env is not None:
                logging.info(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        logging.info("Finished init ENV.")


def get_default_runner(
    seeds: Optional[list] = None,
    device: str = "cuda:0",
    dump_dir: str = None,
) -> InferenceRunner:
    configs_base["use_deepspeed_evo_attention"] = (
        os.environ.get("USE_DEEPSPEED_EVO_ATTTENTION", False) == "true"
    )
    configs_base["model"]["N_cycle"] = 10
    configs_base["sample_diffusion"]["N_sample"] = 1
    configs_base["sample_diffusion"]["N_step"] = 200
    configs_base["device"] = device

    if dump_dir:
        inference_configs["dump_dir"] = dump_dir

    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        fill_required_with_null=True,
    )
    if seeds is not None:
        configs.seeds = seeds
    download_infercence_cache(configs, model_version="v0.2.0")
    return SingleGPUInferenceRunner(configs)


def convert_seq_to_json(task: PredictTask) -> dict:
    """
    Convert task.seq to the specified JSON format.

    Args:
        task (PredictTask): The task containing the sequence and UUID.

    Returns:
        dict: The JSON structure as specified.
    """
    return [
        {
            "sequences": [{"proteinChain": {"sequence": task.seq, "count": 1}}],
            "name": task.id,
        }
    ]


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


def cif_to_pdb(cif_file_path: str, pdb_file_path: str):
    """
    Convert a CIF file to a PDB file, preserving B-factor information.

    Args:
        cif_file_path (str): Path to the input CIF file.
        pdb_file_path (str): Path to the output PDB file.
        scale_b_factors (bool): Whether to scale B-factors by 1/100 (default: True).
    """
    # Read the CIF file
    cif_file = pdbx.CIFFile.read(cif_file_path)
    atom_array = pdbx.get_structure(cif_file, model=1)

    # ============ Migrate B-factor ============
    # Extract B-factor information from the CIF file
    atom_site = cif_file.block["atom_site"]
    if "B_iso_or_equiv" in atom_site:
        # Convert B-factors to a numpy array of float type
        b_factors = np.array(atom_site["B_iso_or_equiv"].as_array(), dtype=float)
    else:
        # If B-factor information is missing, set it to 0
        b_factors = np.zeros(atom_array.array_length(), dtype=float)

    # Scale B-factors if necessary
    b_factors = b_factors / 100.0  # Scale by 1/100

    atom_array.add_annotation("b_factor", dtype=float)
    atom_array.b_factor = b_factors  # Set B-factor information

    # ============ Fix Chain ID ============
    # Create a mapping from multi-character chain IDs to single-character chain IDs
    unique_chain_ids = set(atom_array.chain_id)
    chain_id_mapping = {}
    single_char_chain_ids = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )
    for i, chain_id in enumerate(unique_chain_ids):
        if i >= len(single_char_chain_ids):
            raise ValueError("Too many chains to map to single-character chain IDs.")
        chain_id_mapping[chain_id] = single_char_chain_ids[i]

    # Apply the mapping to the chain IDs
    new_chain_ids = [chain_id_mapping[chain_id] for chain_id in atom_array.chain_id]
    atom_array.chain_id = new_chain_ids

    # ============ Save PDB ============
    # Write the AtomArray to a PDB file
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(atom_array)
    pdb_file.write(pdb_file_path)
