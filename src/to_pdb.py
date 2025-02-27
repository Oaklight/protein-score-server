import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import requests
import yaml
from tqdm import tqdm

tqdm.pandas()

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
config_path = os.path.join(parent_path, "server.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


def extract_name_seq_map(parquet_path: str) -> Dict[str, str]:
    """
    Extracts the name and sequence mapping from the Parquet file.

    Parameters:
    parquet_path (str): Path to the Parquet file containing the data.

    Returns:
    Dict[str, str]: A dictionary mapping each name to its corresponding sequence.
    """
    df = pd.read_parquet(parquet_path)
    return {row["name"]: row["seq"] for _, row in df.iterrows()}


def get_sequence_by_name(name_seq_map: Dict[str, str], name: str) -> str | None:
    """
    Returns the sequence corresponding to the given name.

    Parameters:
    name_seq_map (Dict[str, str]): The dictionary mapping names to sequences.
    name (str): The name for which to find the sequence.

    Returns:
    str | None: The corresponding sequence if found, otherwise None.
    """
    return name_seq_map.get(name)


def get_sequence_by_name_from_pdbbank(name: str) -> str | None:
    """
    Returns the sequence corresponding to the given name by querying pdb bank.

    Args:
        name (str): The name in the format 'protein_id.chain_id' (e.g., '1dgw.Y').

    Returns:
        str | None: The sequence corresponding to the given chain, or None if not found.
    """
    try:
        # Split the name into protein ID and chain ID
        protein_id, chain_id = name.split(".")

        # Fetch the FASTA content from the RCSB PDB database
        url = f"https://www.rcsb.org/fasta/entry/{protein_id}/display"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the FASTA content using Biopython
        fasta_content = response.text
        fasta_io = StringIO(fasta_content)
        for record in SeqIO.parse(fasta_io, "fasta"):
            # Extract the second segment (Chain segment) from the description
            segments = record.description.split("|")
            if len(segments) >= 2 and chain_id in segments[1]:
                return str(record.seq)

        # If the chain ID is not found, return None
        return None

    except Exception as e:
        print(f"Error fetching sequence: {e}")
        return None


def get_name_by_sequence(name_seq_map: Dict[str, str], sequence: str) -> str | None:
    """
    Returns the name corresponding to the given sequence.

    Parameters:
    name_seq_map (Dict[str, str]): The dictionary mapping names to sequences.
    sequence (str): The sequence for which to find the name.

    Returns:
    str | None: The corresponding name if found, otherwise None.
    """
    reverse_map = {seq: name for name, seq in name_seq_map.items()}
    return reverse_map.get(sequence)


def reconstruct_backbone_pdb(
    parquet_path: str,
    output_dir: str,
    pdb_executor: ThreadPoolExecutor,
    atom_wanted: List[Literal["CA", "N", "C", "O"]] = ["CA"],
    existing_pdb: Dict[str, str] = None,
    root_dir=parent_path,
) -> Dict[str, str]:
    """
    Reconstructs PDB files from a Parquet file containing structure information,
    writing them to the specified output directory. Skips structures that already
    have corresponding PDB files in the provided map.

    Parameters:
    parquet_path (str): The filesystem path to the Parquet file.
    output_dir (str): The directory where the PDB files should be output.
    pdb_executor (ThreadPoolExecutor): Executor to provide a pool of threads to run.
    atom_wanted (List[Literal["CA", "N", "C", "O"]]): The specific atoms to
        include when reconstructing the backbone PDB structure. Defaults to only
        include "CA" (Alpha Carbon) if not provided.
    existing_pdb (Dict[str, str]): A pre-existing mapping of names to filesystem
        paths for structures that have already been reconstructed to PDB files.
        Defaults to None.
    root_dir (str): The root filesystem path, used to construct relative paths in
        the output. Defaults to the parent of the current script's directory.

    Returns:
    Dict[str, str]: A mapping of structure names to relative paths of the output PDB
        files, including structures output during this call and those included in
        the existing_pdb mapping.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_parquet(parquet_path)
    # skip those matching rows with existing pdb files, match column "name"
    if existing_pdb:
        df = df[~df["name"].isin(existing_pdb)]

    reversed_index = {
        each_name: os.path.relpath(each_path, root_dir)
        for each_name, each_path in existing_pdb.items()
    }  # reversed index for name to path

    _write_pdb = partial(write_pdb, output_dir=output_dir, atom_wanted=atom_wanted)

    for each_name, each_path in tqdm(
        pdb_executor.map(_write_pdb, df.to_dict(orient="records"), chunksize=20),
        total=len(df),
    ):
        reversed_index[each_name] = os.path.relpath(each_path, root_dir)

    print(f"Done writing pdb files to {output_dir}")

    return reversed_index


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
        residue_data = []
        for atom in atom_wanted:
            coords = data["coords"][atom][i]
            if not any(np.isnan(coords)):
                residue_data.append((atom, coords))

        # write atoms if the residue has at least one desired atom
        for atom, coords in residue_data:
            x, y, z = coords
            if atom == "CA":
                atom_ = "C"
            else:
                atom_ = atom
            bulk_write += f"ATOM  {atom_count:>5d}  {atom:<2s}  ALA A{residue_count:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_}\n"
            atom_count += 1
        residue_count += 1

    output_file = os.path.join(output_dir, f"{data['name']}.pdb")
    with open(output_file, "w") as f:  # remember to close the file
        f.write(bulk_write)

    return (data["name"], output_file)


def get_pdb_file(reversed_index: Dict[str, str], name: str) -> str:
    try:
        pdb_path = reversed_index[name]
        os.path.exists(pdb_path)
    except FileNotFoundError or KeyError as e:
        raise e(f"{name} not found")

    return pdb_path


def process(force=False):
    pdb_executor = ThreadPoolExecutor(max_workers=10)

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(current_path)

    # define paths
    base_path = os.path.join(parent_path, config["backbone_pdb"]["parquet_prefix"])
    output_dir = os.path.join(parent_path, config["backbone_pdb"]["pdb_prefix"])
    index_path = os.path.join(parent_path, config["backbone_pdb"]["reversed_index"])
    name_seq_map_path = os.path.join(
        parent_path, config["backbone_pdb"]["name_seq_map"]
    )

    name_seq_map = {}
    reversed_index = {}
    for version in ["4.3", "4.2"]:
        for split in ["validation", "test"]:

            # get existing file names in the output directory in case of restart. Stem to only get the file name without extension and base path
            if force:
                existing_pdb = None
            else:
                existing_pdb = {
                    os.path.splitext(os.path.basename(f))[0]: f
                    for f in glob.glob(f"{output_dir}{version}/*.pdb")
                }

            name_seq_map.update(
                extract_name_seq_map(
                    parquet_path=f"{base_path}{version}/{split}.parquet"
                )
            )
            reversed_index.update(
                reconstruct_backbone_pdb(
                    parquet_path=f"{base_path}{version}/{split}.parquet",
                    output_dir=f"{output_dir}{version}",
                    pdb_executor=pdb_executor,
                    atom_wanted=["N", "CA", "C", "O"],
                    existing_pdb=existing_pdb,
                )
            )

    # use relative path
    reversed_index = {
        key: os.path.relpath(value) for key, value in reversed_index.items()
    }

    # save reversed index to the same folder of output_dir, rip off the final part from output_dir
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            # update the file with the new reversed index
            existing_index = yaml.safe_load(f) or {}
        reversed_index.update(existing_index)
    with open(index_path, "w") as f:
        yaml.dump(reversed_index, f)

    # save name_seq_map to the same folder of output_dir, rip off the final part from output_dir
    if os.path.exists(name_seq_map_path):
        with open(name_seq_map_path, "r") as f:
            # update the file with the new name_seq_map
            existing_name_seq_map = yaml.safe_load(f) or {}
        name_seq_map.update(existing_name_seq_map)
    with open(name_seq_map_path, "w") as f:
        yaml.dump(name_seq_map, f)

    pdb_executor.shutdown()

    print(f"Saved reversed index to {index_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--force", action="store_true", help="Force overwrite existing pdb files"
    )
    args = args.parse_args()
    process(args.force)
