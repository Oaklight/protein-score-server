from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", low_cpu_mem_usage=True
)

model = model.cuda()
model.esm = model.esm.half()
import torch

torch.backends.cuda.matmul.allow_tf32 = True
# Uncomment this line if your GPU memory is 16GB or less, or if you're folding longer (over 600 or so) sequences
model.trunk.set_chunk_size(64)


# This is the sequence for human GNAT1, because I worked on it when
# I was a postdoc and so everyone else has to learn to appreciate it too.
# Feel free to substitute your own peptides of interest
# Depending on memory constraints you may wish to use shorter sequences.
test_protein = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"

tokenized_input = tokenizer(
    [test_protein], return_tensors="pt", add_special_tokens=False
)["input_ids"]

tokenized_input = tokenized_input

import torch

with torch.no_grad():
    output = model(tokenized_input)


from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
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
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


pdb = convert_outputs_to_pdb(output)
# print(pdb)

with open("result_esmfold.pdb", "w") as f:
    f.write(pdb[0])

import biotite.structure.io as bsio

struct = bsio.load_structure("result_esmfold.pdb", extra_fields=["b_factor"])

print(struct.b_factor.mean())  # this will be the pLDDT

# 0.9505201282508016
# 0.9505201282508016
# 0.9505201282508016
