# generate_graph.py

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any

from prune_graph import save_baseline, validate_thresholds

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google/gemma-3-1b-pt"
MODEL_REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"
CLT_ID = "google/gemma-scope-2-1b-pt"
CLT_REVISION = "b738dc06961818c011fb2e44a316352ca0f4e873"
CLT_FOLDER = "clt/width_262k_l0_medium_affine"
TRACER_REVISION = "255d989e93be1299d53d67b59e9fd76a0e274b14"
PROMPTS = ROOT / "datasets/prepared/time-course-mechinterp/prompts.jsonl"


def clt_config() -> dict[str, Any]:
    shards = [
        f"hf://{CLT_ID}/{CLT_FOLDER}/params_layer_{i}.safetensors?revision={CLT_REVISION}"
        for i in range(26)
    ]
    # At this tracer commit the loader iterates range(max(paths.keys())). A
    # duplicate final path is a sentinel: 26 real layers, not a 27-layer model.
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


def choose_prompt(prompt: str | None, row: int, path: Path = PROMPTS) -> dict[str, Any]:
    if prompt is not None:
        if not prompt.strip() or len(prompt) > 256:
            raise ValueError("Use a nonempty prompt of at most 256 characters.")
        return {"prompt_prefix": prompt, "answer": None, "source": "custom prompt"}
    if row < 0:
        raise ValueError("--row must be nonnegative.")
    if not path.is_file():
        raise FileNotFoundError("Run code/example_code/prepare_data.py first.")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if row >= len(records):
        raise ValueError(f"--row must be below {len(records)} for this prompt file.")
    record = records[row]
    if not record["prompt_prefix"].strip() or len(record["prompt_prefix"]) > 256:
        raise ValueError("Prepared prompt must contain 1..256 characters.")
    return record


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run(
    record: dict[str, Any], output: Path, max_features: int = 128,
    node_threshold: float = 0.8, edge_threshold: float = 0.98,
) -> dict[str, Any]:
    # Thresholds set cumulative influence cutoffs for node and edge pruning.
    validate_thresholds(node_threshold, edge_threshold)
    if not 1 <= max_features <= 128:
        raise ValueError("Keep this baseline small: --max-features must be 1..128.")
    choose_prompt(record["prompt_prefix"], 0)  # Validate before any downloads.
    output.mkdir(parents=True, exist_ok=False)  # Preserve previous run artifacts.
    manifest = {
        "status": "started",
        "stage": "dependencies",
        "record": record,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "clt": clt_config(),
        "tracer_revision": TRACER_REVISION,
        "backend": "nnsight",
        "dtype": "bfloat16",
        "max_features": max_features,
        "pruning": {"method": "circuit_tracer.graph.prune_graph", "node_threshold": node_threshold,
                    "edge_threshold": edge_threshold},
    }
    start = time.perf_counter()
    _write_manifest(output, manifest)
    try:
        # Set finite HTTP timeouts before importing the optional HF dependency.
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
        os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
            from circuit_tracer import ReplacementModel, attribute
            from circuit_tracer.utils.hf_utils import load_transcoders
            from circuit_tracer.utils.create_graph_files import create_graph_files
        except ImportError as exc:
            raise RuntimeError(
                "Install graph dependencies with: uv sync --locked --extra graph"
            ) from exc

        dist = importlib.metadata.distribution("circuit-tracer")
        receipt = json.loads(dist.read_text("direct_url.json") or "{}")
        if receipt.get("vcs_info", {}).get("commit_id") != TRACER_REVISION:
            raise RuntimeError(
                "circuit-tracer differs from the audited pin; run uv sync --locked --extra graph."
            )
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            raise RuntimeError(
                "Use a CUDA GPU with native BF16 support (for example A100), plus high host RAM."
            )
        manifest["gpu"] = torch.cuda.get_device_name()
        manifest["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        manifest["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "nnsight", "circuit-tracer")
        }
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        manifest["stage"] = "model_download"
        _write_manifest(output, manifest)
        snapshot = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            max_workers=2,
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        tokens = tokenizer(record["prompt_prefix"], add_special_tokens=True)["input_ids"]
        if len(tokens) > 32:
            raise ValueError("Shorten the prompt to at most 32 tokens, including BOS.")
        manifest["prompt_token_ids"] = tokens
        manifest["stage"] = "clt_download_and_load"
        _write_manifest(output, manifest)
        # Raw Scope 2 loading temporarily holds two full weight copies. Stage on
        # CPU to avoid that peak on the GPU; this still needs substantial host RAM.
        clt = load_transcoders(
            clt_config(),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            lazy_encoder=False,
            lazy_decoder=False,
        )
        if clt.n_layers != 26:
            raise ValueError("Expected all 26 CLT layers; recheck the pinned loader adapter.")
        # The NNsight constructor ignores revision kwargs, so give it the pinned
        # local snapshot. It also moves the CLT (including affine skip) to BF16.
        model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=str(Path(snapshot).resolve()),
            transcoders=clt,
            backend="nnsight",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        manifest["stage"] = "attribution"
        _write_manifest(output, manifest)
        graph = attribute(
            tokens,
            model,
            max_feature_nodes=max_features,
            batch_size=1,
            max_n_logits=2,
            desired_logit_prob=0.95,
            verbose=True,
        )
        matrix = graph.adjacency_matrix
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not torch.isfinite(matrix).all()
        ):
            raise ValueError("Expected a finite, square attribution matrix.")
        if graph.selected_features.numel() > max_features:
            raise ValueError("The graph exceeded the requested feature cap.")
        manifest["graph"] = {
            "nodes": matrix.shape[0],
            "selected_features": graph.selected_features.numel(),
            "nonzero_edges": torch.count_nonzero(matrix).item(),
            "adjacency_convention": "row=target,column=source",
            "logit_token_ids": graph.logit_token_ids.tolist(),
            "logit_tokens": [tokenizer.decode([i]) for i in graph.logit_token_ids.tolist()],
            "logit_probabilities": graph.logit_probabilities.tolist(),
        }
        graph.to("cpu")
        graph.to_pt(str(output / "graph.pt"))
        manifest["stage"] = "baseline_pruning"
        _write_manifest(output, manifest)
        # Keep graph.pt unchanged so later methods use the same candidate graph.
        manifest["baseline"] = save_baseline(
            graph, output / "baseline", node_threshold, edge_threshold,
        )
        manifest["stage"] = "viewer_export"
        _write_manifest(output, manifest)
        # Use the same native thresholds for the viewer and saved baseline masks.
        create_graph_files(
            graph,
            slug="gemma-scope2-baseline",
            output_path=str(output / "viewer"),
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
        manifest["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest["status"], manifest["stage"] = "complete", "complete"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - start
        _write_manifest(output, manifest)
    return manifest


# main: Select a factual prompt and run one graph, or inspect its setup offline.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(
        "Generate one capped Gemma Scope 2 CLT attribution graph using circuit-tracer.\n"
        "\n"
        "Run from the repository: python code/example_code/generate_graph.py --dry-run\n"
        "The real run needs a CUDA GPU, large host RAM, and Hugging Face Gemma access.\n"
    ))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--row", type=int, default=0, help="Prepared prompt index (default 0).")
    selection.add_argument("--prompt", help="Use this short prompt instead of a dataset row.")
    parser.add_argument("--output", type=Path, default=ROOT / "code/example_code/output/baseline")
    parser.add_argument(
        "--max-features", type=int, choices=range(1, 129), default=128, metavar="1..128"
    )
    parser.add_argument("--node-threshold", type=float, default=0.8,
                        help="Native cumulative node influence cutoff (0, 1].")
    parser.add_argument("--edge-threshold", type=float, default=0.98,
                        help="Native cumulative edge influence cutoff (0, 1].")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print setup; no model imports/downloads."
    )
    args = parser.parse_args(argv)
    try:
        validate_thresholds(args.node_threshold, args.edge_threshold)
        record = choose_prompt(args.prompt, args.row)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "record": record,
                        "model": MODEL_ID,
                        "model_revision": MODEL_REVISION,
                        "clt": clt_config(),
                        "max_features": args.max_features,
                        "pruning": {"method": "circuit_tracer.graph.prune_graph",
                                    "node_threshold": args.node_threshold,
                                    "edge_threshold": args.edge_threshold},
                    },
                    indent=2,
                )
            )
        else:
            run(record, args.output, args.max_features, args.node_threshold, args.edge_threshold)
            print(f"Saved graph.pt, baseline/, viewer/, and manifest.json in {args.output}")
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
