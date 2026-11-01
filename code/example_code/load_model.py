
"""Load Gemma 3 1B and its Gemma Scope 2 CLT checkpoint."""



from __future__ import annotations



import argparse

import importlib.metadata

import json

import os

from pathlib import Path

from typing import Any



ROOT = Path(__file__).resolve().parents[2]



MODEL_ID = "google/gemma-3-1b-pt"

MODEL_REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"



CLT_ID = "google/gemma-scope-2-1b-pt"

CLT_REVISION = "b738dc06961818c011fb2e44a316352ca0f4e873"

CLT_FOLDER = "clt/width_262k_l0_medium_affine"



TRACER_REVISION = "255d989e93be1299d53d67b59e9fd76a0e274b14"





def clt_config() -> dict[str, Any]:

    """Return the pinned 26-layer CLT configuration."""

    shards = [

        f"hf://{CLT_ID}/{CLT_FOLDER}/params_layer_{i}.safetensors"

        f"?revision={CLT_REVISION}"

        for i in range(26)

    ]



    # Sentinel required by the pinned circuit-tracer loader.

    shards.append(shards[-1])



    return {

        "model_name": MODEL_ID,

        "model_kind": "cross_layer_transcoder",

        "repo_id": CLT_ID,

        "revision": CLT_REVISION,

        "feature_input_hook": "mlp.hook_in",

        "feature_output_hook": "hook_mlp_out",

        "scan_name": f"{CLT_ID}/{CLT_FOLDER}@{CLT_REVISION}",

        "transcoders": shards,

    }





def load_model(cache_dir: Path):

    """Load Gemma 3 1B and integrate its CLT checkpoint."""



    # Set the Hugging Face cache before importing HF-dependent libraries.

    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")



    import torch

    from huggingface_hub import snapshot_download

    from circuit_tracer import ReplacementModel

    from circuit_tracer.utils.hf_utils import load_transcoders



    # Verify the pinned circuit-tracer installation.

    dist = importlib.metadata.distribution("circuit-tracer")

    receipt = json.loads(dist.read_text("direct_url.json") or "{}")



    actual_revision = receipt.get("vcs_info", {}).get("commit_id")

    if actual_revision != TRACER_REVISION:

        raise RuntimeError(

            f"Expected circuit-tracer {TRACER_REVISION}, "

            f"but found {actual_revision}."

        )



    # Verify GPU and native BF16 support.

    if not torch.cuda.is_available():

        raise RuntimeError("CUDA GPU is not available.")



    if not torch.cuda.is_bf16_supported(including_emulation=False):

        raise RuntimeError("A GPU with native BF16 support is required.")



    device = torch.device("cuda")



    print("GPU:", torch.cuda.get_device_name(0))

    print(

        "GPU memory (GB):",

        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),

    )



    # Download or reuse the exact pinned base-model snapshot.

    print("\n[1/3] Preparing Gemma 3 1B checkpoint...")



    snapshot = snapshot_download(

        repo_id=MODEL_ID,

        revision=MODEL_REVISION,

        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],

        max_workers=2,

    )



    print("Base model snapshot:", snapshot)



    # Load the complete CLT checkpoint into CPU memory first.

    print("\n[2/3] Loading the 26-layer CLT checkpoint into CPU memory...")



    clt = load_transcoders(

        clt_config(),

        device=torch.device("cpu"),

        dtype=torch.bfloat16,

        lazy_encoder=False,

        lazy_decoder=False,

    )



    if clt.n_layers != 26:

        raise RuntimeError(

            f"Expected 26 CLT layers, found {clt.n_layers}."

        )



    print("CLT loaded. Number of layers:", clt.n_layers)



    # Integrate the CLT with the original Gemma model.

    print("\n[3/3] Integrating Gemma 3 1B and CLT...")



    model = ReplacementModel.from_pretrained_and_transcoders(

        model_name=str(Path(snapshot).resolve()),

        transcoders=clt,

        backend="nnsight",

        device=device,

        dtype=torch.bfloat16,

    )






    print("\n===== LOADING SUCCESSFUL =====")

    print("Base model:", MODEL_ID)

    print("Base model revision:", MODEL_REVISION)

    print("CLT:", CLT_ID)

    print("CLT revision:", CLT_REVISION)

    print("CLT layers:", clt.n_layers)

    print("Backend: nnsight")

    print("Device:", device)

    print("Data type: bfloat16")

    print(

        "GPU memory allocated (GB):",

        round(torch.cuda.memory_allocated() / 1024**3, 2),

    )

    print(

        "GPU peak memory (GB):",

        round(torch.cuda.max_memory_allocated() / 1024**3, 2),

    )



    return model





def main():

    parser = argparse.ArgumentParser(

        description="Load Gemma 3 1B and its Gemma Scope 2 CLT checkpoint."

    )

    parser.add_argument(

        "--cache-dir",

        type=Path,

        default=ROOT / ".cache/huggingface",

        help="Hugging Face cache directory.",

    )



    args = parser.parse_args()

    load_model(args.cache_dir)





if __name__ == "__main__":

    main()

