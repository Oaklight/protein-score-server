import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

# Import for protenix
from runner.inference import infer_predict
from runner.msa_search import contain_msa_res, msa_search_update

from src.task import PredictTask
from src.model import (
    get_default_runner,
    cif_to_pdb,
    convert_seq_to_json,
)


def main(device_id: int = 0):
    result_dir = "./protenix_test_dir"
    # Copy the result from output_temp_dir to result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    task = PredictTask(
        seq="MAEVIRSSAFWRSFPIFEEFDSETLCELSGIASYRKWSAGTVIFQRGDQGDYMIVVVSGRIKLSLFTPQGRELMLRQHEAGALFGEMALLDGQPRSADATAVTAAEGYVIGKKDFLALITQRPKTAEAVIRFLCAQLRDTTDRLETIALYDLNARVARFFLATLRQIHGSEMPQSANLRLTLSQTDIASILGASRPKVNRAILSLEESGAIKRADGIICCNVGRLLSIADPEEDLEHHHHHHHH",
        task_type="plddt",
    )

    seeds = [101]
    temp_outdir = "./test_protenix"

    input_json_temp_dir = os.path.join(temp_outdir, "input_json")
    os.makedirs(input_json_temp_dir, exist_ok=True)
    output_temp_dir = os.path.join(temp_outdir, "output")
    os.makedirs(output_temp_dir, exist_ok=True)

    infer_json = os.path.join(input_json_temp_dir, f"input_{task.id}.json")
    # save json to temp_dir
    with open(infer_json, "w") as f:
        json.dump(convert_seq_to_json(task), f)

    model = get_default_runner(
        seeds=seeds, device=f"cuda:{device_id}", dump_dir=output_temp_dir
    )

    configs = model.configs

    infer_json = msa_search_update(infer_json, input_json_temp_dir)

    configs["input_json_path"] = infer_json
    if not contain_msa_res(infer_json):
        raise RuntimeError(
            f"`{infer_json}` has no msa result for `proteinChain`, please add first."
        )
    infer_predict(model, configs)

    # result dump to
    protenix_cif_path = os.path.join(
        output_temp_dir,
        task.id,
        f"seed_{configs['seeds'][0]}",
        "predictions",
        f"{task.id}_seed_{configs['seeds'][0]}_sample_0.cif",
    )
    protenix_pdb_path = os.path.join(output_temp_dir, f"test_{task.id}.pdb")

    print(f"[{task.id}] writing pdb to {protenix_pdb_path}")
    cif_to_pdb(protenix_cif_path, protenix_pdb_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Protenix inference.")
    parser.add_argument(
        "--device_id", type=int, default=0, help="CUDA device ID to use."
    )
    args = parser.parse_args()
    device_id = args.device_id
    main(device_id)
