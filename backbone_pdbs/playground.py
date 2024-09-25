from time import time

import biotite.structure.io as bsio
from huggingface_hub import login
from TMscore import TMscore

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

pdb_origin = "pdb/1A1X.pdb"
pdb_backbone = "pdb/1A1X_backbone.pdb"
pdb_backbone_2 = "pdb/1A1X_backbone_2.pdb"
pdb_esm = "pdb/1A1X_esm.pdb"
pdb_rebuilt = "pdb/1A1X_rebuilt.pdb"
pdb_rebuilt_pre = "pdb/protein_backbone_cath_4.2/1a1x.A.pdb"

seq = "GSAGEDVGAPPDHLWVHQEGIYRDEYQRTWVAVVEEETSFLRARVQQIQVPLGDAARPSHLLTSQLPLMWQLYPEERYMDNNSRLWQIQHHLMVRGVQELLLKLLPDD"


def strip_pdb():
    pdb_lines_backbone = []
    pdb_lines = []
    with open(pdb_origin, "r") as f:
        lines = f.readlines()

    for line in lines:
        elements = line.split()
        if elements[0] != "ATOM":
            continue
        pdb_lines.append(line)
        if elements[2] not in ["CA", "N", "C", "O"]:
            continue
        pdb_lines_backbone.append(line)

    with open(pdb_origin, "w") as f:
        f.writelines(pdb_lines)

    with open(pdb_backbone, "w") as f:
        f.writelines(pdb_lines_backbone)


def calc_tmscore(pdb_1, pdb_2):

    lengths, results = TMscore(pdb_1, pdb_2)

    print(lengths)
    print(results)


def rebuild_pdb():
    pdb_lines_rebuilt = []
    pdb_lines = []
    with open(pdb_origin, "r") as f:
        lines = f.readlines()

    residue_count = 0
    atom_count = 0
    last_residue = ""

    for line in lines:
        elements = line.split()
        if elements[0] != "ATOM":
            continue

        atom = elements[2]
        residue_name = elements[5]
        if residue_name != last_residue:
            residue_count += 1
            last_residue = residue_name

        x, y, z = float(elements[6]), float(elements[7]), float(elements[8])

        if elements[0] != "ATOM":
            continue
        pdb_lines.append(line)
        if elements[2] not in ["CA", "N", "C", "O"]:
            continue

        if atom == "CA":
            atom_ = "C"
        elif atom == "N":
            atom_ = "N"
        elif atom == "C":
            atom_ = "C"
        elif atom == "O":
            atom_ = "O"

        atom_count += 1

        pdb_lines_rebuilt.append(
            f"ATOM  {atom_count:>5d}  {atom:<2s}  ALA A{residue_count:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_}\n"
        )

    with open(pdb_rebuilt, "w") as f:
        f.writelines(pdb_lines_rebuilt)


def esm_fold(seq, num_steps=5):

    # Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
    login("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")

    # This will download the model weights and instantiate the model on your machine.
    model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

    # Generate a completion for a partial Carbonic Anhydrase (2vvb)
    protein = ESMProtein(sequence=seq)

    protein = model.generate(
        protein, GenerationConfig(track="structure", num_steps=num_steps)
    )

    protein.to_pdb(pdb_esm)

    struct = bsio.load_structure(pdb_esm, extra_fields=["b_factor"])

    print(struct.b_factor.mean())  # this will be the pLDDT


def modify_residue_sequence_number(input_file, output_file):
    """
    Modifies the 5th column (residue sequence number) in a PDB file so that it starts from 1 instead of 4.
    Args:
    input_file (str): The name of the input PDB file.
    output_file (str): The name of the output PDB file with modified residue sequence numbers.
    """
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Only modify lines that start with "ATOM"
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Extract the fields from the line
                atom_serial = line[:6]
                atom_name = line[6:11]
                residue_name = line[17:20]
                chain_id = line[21]
                residue_seq = int(line[22:26])  # Get the residue sequence number
                # Adjust the residue sequence number to start from 1
                new_residue_seq = residue_seq - 3
                # Write the modified line to the output file
                outfile.write(
                    f"{atom_serial}{atom_name}{line[11:17]}{residue_name} {chain_id}{new_residue_seq:4d}{line[26:]}"
                )
            else:
                # Write lines that do not start with "ATOM" as they are
                outfile.write(line)


if __name__ == "__main__":

    strip_pdb()
    rebuild_pdb()
    modify_residue_sequence_number(pdb_backbone, pdb_backbone_2)
    esm_fold(seq, 20)

    print(f"pdb_origin vs pdb_backbone")
    calc_tmscore(pdb_origin, pdb_backbone)
    print()

    print(f"pdb_backbone vs pdb_backbone_2")
    calc_tmscore(pdb_backbone, pdb_backbone_2)
    print()

    print(f"pdb_backbone vs pdb_rebuilt")
    calc_tmscore(pdb_backbone, pdb_rebuilt)
    print()

    print(f"pdb_backbone_2 vs pdb_rebuilt")
    calc_tmscore(pdb_backbone_2, pdb_rebuilt)
    print()

    print(f"pdb_rebuilt vs pdb_rebuilt_pre")
    calc_tmscore(pdb_rebuilt, pdb_rebuilt_pre)
    print()

    print(f"pdb_backbone vs pdb_esm")
    calc_tmscore(pdb_backbone, pdb_esm)
    print()

    print(f"pdb_backbone_2 vs pdb_esm")
    calc_tmscore(pdb_backbone_2, pdb_esm)
    print()

    print(f"pdb_rebuilt vs pdb_esm")
    calc_tmscore(pdb_rebuilt, pdb_esm)
    print()

    print(f"pdb_rebuilt_pre vs pdb_esm")
    calc_tmscore(pdb_rebuilt_pre, pdb_esm)
    print()
