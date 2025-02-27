import requests


def get_sequence_by_name_from_pdbbank_1(name: str) -> str | None:
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

        # Parse the FASTA content to find the sequence for the given chain
        fasta_content = response.text
        for entry in fasta_content.split(">")[1:]:
            header, sequence = entry.split("\n", 1)
            # Extract the second segment (Chain segment) from the header
            segments = header.split("|")
            if len(segments) >= 2 and chain_id in segments[1]:
                return sequence.replace("\n", "")

        # If the chain ID is not found, return None
        return None

    except Exception as e:
        print(f"Error fetching sequence: {e}")
        return None


import requests
from io import StringIO
from Bio import SeqIO


def get_sequence_by_name_from_pdbbank_2(name: str) -> str | None:
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


# Example usage:
# sequence = get_sequence_by_name_from_pdbbank('1dgw.Y')
# print(sequence)  # Output: "MQLRRYAATLSEGDIIVIPSSFPVALKAASDLNMVGIGVNAENNERNFLAGHKENVIRQIPRQVSDLTFPGSGEEVEELLENQKESYFVDGQP"

if __name__ == "__main__":
    test = "1dgw.Y"
    test = "1cxz.B"
    seq1 = get_sequence_by_name_from_pdbbank_1(test)
    seq2 = get_sequence_by_name_from_pdbbank_2(test)
    print(seq1)
    print()
    print(seq2)
