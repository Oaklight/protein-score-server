from biotite.structure.io import pdbx, pdb
import numpy as np


def cif_to_pdb(
    cif_file_path: str, pdb_file_path: str, scale_b_factors: bool = True
) -> None:
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


def print_cif_b_factors(cif_file_path: str) -> None:
    """
    Print the B-factor column from a CIF file.

    Args:
        cif_file_path (str): Path to the input CIF file.
    """
    # Read the CIF file
    cif_file = pdbx.CIFFile.read(cif_file_path)

    # Extract the atom_site category, which contains B-factor information
    atom_site = cif_file.block["atom_site"]

    # Check if B-factor information is present
    if "B_iso_or_equiv" in atom_site:
        b_factors = atom_site["B_iso_or_equiv"].as_array()
        print("B-factor values from CIF file:")
        for i, b_factor in enumerate(b_factors):
            print(f"Atom {i+1}: {b_factor}")
    else:
        print("No B-factor information found in the CIF file.")
