import os
import sys

import yaml
from TMscore import TMscore

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src import to_pdb


task_name = "12as.A"

reversed_index_path = "../backbone_pdbs/pdb/reversed_index.yaml"

with open(reversed_index_path, "r") as f:
    reversed_index = yaml.safe_load(f)

reference_pdb = to_pdb.get_pdb_file(reversed_index, task_name)
temp_pdb_file = "../intermediate_pdbs/result_huggingface_esmfold_7af7055db1eb46f28883f0a7b0e4d41b.pdb"

lengths, results = TMscore(reference_pdb, temp_pdb_file)

output_data = results[0]
