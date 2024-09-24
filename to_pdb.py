import glob
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Literal

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

tqdm.pandas()

current_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_path, "server.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


def reconstruct_backbone_pdb(
    parquet_path: str,
    output_dir: str,
    atom_wanted: List[Literal["CA", "N", "C", "O"]] = ["CA"],
    existing_pdb: List[str] = None,
):
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    df = pd.read_parquet(parquet_path)
    # skip those matching rows with existing pdb files, match column "name"
    if existing_pdb is not None:
        df = df[~df["name"].isin(existing_pdb)]

    executor = ThreadPoolExecutor(max_workers=10)

    _write_pdb = partial(write_pdb, output_dir=output_dir, atom_wanted=atom_wanted)

    for _ in tqdm(
        executor.map(_write_pdb, df.to_dict(orient="records"), chunksize=20),
        total=len(df),
    ):
        pass

    print(f"Done writing pdb files to {output_dir}")
    executor.shutdown()


def write_pdb(
    data,
    output_dir,
    atom_wanted: List[Literal["CA", "N", "C", "O"]] = ["CA"],
):
    # the assumption is that we only know of the atom and its coordinates, nothing else.
    # Thus the other information in a standard pdb file is left as templates.
    atom_count = 1
    residue_count = 1

    bulk_write = ""
    for i in range(len(data["seq"])):
        # for each residue, check if desired atoms have non-nan coordinates
        residue_data = {}
        for atom in atom_wanted:
            coords = data["coords"][atom][i]
            if not any(np.isnan(coords)):
                residue_data[atom] = coords

        # write atoms if the residue has at least one desired atom
        if len(residue_data) != 0:
            for atom in atom_wanted:
                if atom not in residue_data:
                    continue
                x, y, z = residue_data[atom]
                bulk_write += f"ATOM  {atom_count:>5d}  {atom:<2s}  ALA A{residue_count:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom}\n"
                atom_count += 1
            residue_count += 1

    output_file = os.path.join(output_dir, f"{data['name']}.pdb")
    with open(output_file, "w") as f:  # remember to close the file
        f.write(bulk_write)


def get_pdb_file(name: str) -> str:
    pdb_path = os.path.join(config["backbone_pdb"]["pdb_prefix"], name + ".pdb")

    try:
        os.path.exists(pdb_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"{pdb_path} not found")

    return pdb_path


def process(force=False):
    current_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_path, config["backbone_pdb"]["parquet_prefix"])
    output_dir = os.path.join(current_path, config["backbone_pdb"]["pdb_prefix"])
    for version in ["4.3", "4.2"]:
        for split in ["validation", "test"]:

            # get existing file names in the output directory in case of restart. Stem to only get the file name without extension and base path
            if force:
                existing_pdb = None
            else:
                existing_pdb = [
                    os.path.splitext(os.path.basename(f))[0]
                    for f in glob.glob(f"{output_dir}{version}/*.pdb")
                ]

            reconstruct_backbone_pdb(
                parquet_path=f"{base_path}{version}/{split}.parquet",
                output_dir=f"{output_dir}{version}",
                atom_wanted=["CA", "N", "C", "O"],
                existing_pdb=existing_pdb,
            )


if __name__ == "__main__":
    process()
