from time import time

import biotite.structure.io as bsio
from huggingface_hub import login

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
login("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")

# This will download the model weights and instantiate the model on your machine.
t_loading = time()
model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
t_loading_done = time() - t_loading

# Generate a completion for a partial Carbonic Anhydrase (2vvb)
prompt = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

t_wrap = time()
protein = ESMProtein(sequence=prompt)
t_wrap_done = time() - t_wrap

t_infer = time()
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
t_infer_done = time() - t_infer

t_write = time()
protein.to_pdb("./result_esm3.pdb")
t_write_done = time() - t_write

t_struct_load = time()
struct = bsio.load_structure("result_esm3.pdb", extra_fields=["b_factor"])
t_struct_load_done = time() - t_struct_load

t_plddt = time()
print(struct.b_factor.mean())  # this will be the pLDDT
t_plddt_done = time() - t_plddt

print(f"Total time: {time() - t_loading:.4f}s")
print(f"  - Model loading: {t_loading_done:.4f}s")
print(f"  - Protein wrapping: {t_wrap_done:.4f}s")
print(f"  - Inference: {t_infer_done:.4f}s")
print(f"  - Writing: {t_write_done:.4f}s")
print(f"  - Loading structure: {t_struct_load_done:.4f}s")
print(f"  - pLDDT: {t_plddt_done:.4f}s")


# 0.9574903474903476
# Total time: 6.5826s
#   - Model loading: 2.4120s
#   - Protein wrapping: 0.0000s
#   - Inference: 4.1538s
#   - Writing: 0.0132s
#   - Loading structure: 0.0036s
#   - pLDDT: 0.0001s
