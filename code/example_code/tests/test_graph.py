# # test_graph.py
# Exercise the small runner without model weights, CUDA, or optional packages.
#

import builtins
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import generate_graph as runner


# test_clt_config_pins_every_real_layer_and_loader_sentinel: Check clt config pins every real layer and loader sentinel.
def test_clt_config_pins_every_real_layer_and_loader_sentinel():
    config = runner.clt_config()
    revision = "b738dc06961818c011fb2e44a316352ca0f4e873"
    expected = [
        "hf://google/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine/"
        f"params_layer_{layer}.safetensors?revision={revision}"
        for layer in range(26)
    ]
    assert config["transcoders"] == expected + [expected[-1]]
    assert len(set(config["transcoders"])) == 26
    assert config["model_kind"] == "cross_layer_transcoder"
    assert config["feature_input_hook"] == "mlp.hook_in"
    assert config["feature_output_hook"] == "hook_mlp_out"
    assert config["revision"] == revision
    assert runner.MODEL_ID == "google/gemma-3-1b-pt"
    assert runner.MODEL_REVISION == "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"
    assert runner.TRACER_REVISION == "255d989e93be1299d53d67b59e9fd76a0e274b14"


# test_choose_prompt_rejects_invalid_custom_prompt: Check choose prompt rejects invalid custom prompt.
@pytest.mark.parametrize("prompt", ["", " \n", "x" * 257])
def test_choose_prompt_rejects_invalid_custom_prompt(prompt):
    with pytest.raises(ValueError, match="nonempty prompt"):
        runner.choose_prompt(prompt, 0)


# test_choose_prompt_preserves_custom_text_and_accepts_limit: Check choose prompt preserves custom text and accepts limit.
def test_choose_prompt_preserves_custom_text_and_accepts_limit(tmp_path):
    text = " " + "x" * 255
    assert runner.choose_prompt(text, 0, tmp_path / "absent") == {
        "prompt_prefix": text, "answer": None, "source": "custom prompt"
    }


# test_choose_prompt_selects_rows_ignoring_blank_lines: Check choose prompt selects rows ignoring blank lines.
def test_choose_prompt_selects_rows_ignoring_blank_lines(tmp_path):
    records = [
        {"prompt_prefix": "The capital of France is", "answer": "Paris"},
        {"prompt_prefix": "The capital of Japan is", "answer": "Tokyo"},
    ]
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n" + "\n\n".join(map(json.dumps, records)) + "\n")
    assert runner.choose_prompt(None, 0, path) == records[0]
    assert runner.choose_prompt(None, 1, path) == records[1]
    with pytest.raises(ValueError, match="nonnegative"):
        runner.choose_prompt(None, -1, path)
    with pytest.raises(ValueError, match="below 2"):
        runner.choose_prompt(None, 2, path)
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        runner.choose_prompt(None, 0, tmp_path / "absent")


# test_choose_prompt_rejects_invalid_prepared_text: Check choose prompt rejects invalid prepared text.
@pytest.mark.parametrize("prompt", ["  ", "x" * 257])
def test_choose_prompt_rejects_invalid_prepared_text(tmp_path, prompt):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"prompt_prefix": prompt}) + "\n")
    with pytest.raises(ValueError, match="Prepared prompt"):
        runner.choose_prompt(None, 0, path)


# test_dry_run_imports_no_model_packages_or_creates_output: Check dry run imports no model packages or creates output.
def test_dry_run_imports_no_model_packages_or_creates_output(tmp_path, monkeypatch, capsys):
    original_import = builtins.__import__

    # guarded_import: Reject model-package imports during the dry run.
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "huggingface_hub", "circuit_tracer"}:
            pytest.fail(f"Dry run tried to import optional package {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    # Reload the script too: its top-level imports must remain lightweight.
    spec = importlib.util.spec_from_file_location("offline_graph_runner", runner.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "no-run"
    module.main(["--prompt", "The capital of France is", "--dry-run", "--output", str(output)])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_revision"] == runner.MODEL_REVISION
    assert receipt["clt"] == runner.clt_config()
    assert receipt["record"]["answer"] is None
    assert not output.exists()


# stack: Supply only the external API surface the runner calls, all in memory.
@pytest.fixture
def stack(tmp_path, monkeypatch):
    events = []
    state = SimpleNamespace(tokens=[2, 10, 11], snapshot=tmp_path / "pinned-model")
    state.clt = SimpleNamespace(n_layers=26)
    state.model = object()
    state.matrix = SimpleNamespace(ndim=2, shape=(4, 4), finite=True, nonzero=3)
    state.graph = SimpleNamespace(
        adjacency_matrix=state.matrix, selected_features=Mock(numel=Mock(return_value=2)),
        logit_token_ids=Mock(tolist=Mock(return_value=[12, 13])),
        logit_probabilities=Mock(tolist=Mock(return_value=[0.6, 0.2])),
        to=Mock(), to_pt=Mock(side_effect=lambda path: Path(path).write_text("fake graph")),
    )
    state.tokenizer = Mock(side_effect=lambda *args, **kwargs: {"input_ids": state.tokens})
    state.tokenizer.decode.side_effect = lambda ids: f"token-{ids[0]}"
    state.tokenizer_loader = Mock(return_value=state.tokenizer)

    # record_call: Build a mock that records execution order and returns a fixed result.
    def record_call(event, result):
        # call: Record this stage and return the configured result.
        def call(*args, **kwargs):
            events.append(event)
            return result
        return Mock(side_effect=call)

    state.download = record_call("model_snapshot", str(state.snapshot))
    state.load_clt = record_call("clt_cpu", state.clt)
    state.load_model = record_call("replacement_model", state.model)
    state.attribute = record_call("attribute", state.graph)

    # export: Write a fake viewer export and record its execution order.
    def export(*args, **kwargs):
        events.append("viewer")
        path = Path(kwargs["output_path"])
        path.mkdir()
        (path / "fake.json").write_text("{}")

    state.export = Mock(side_effect=export)

    # prune: Write fake pruning artifacts for orchestration checks.
    def prune(graph, output, node_threshold, edge_threshold):
        events.append("pruning")
        output.mkdir()
        summary = {"method": "native_influence",
                   "thresholds": {"node": node_threshold, "edge": edge_threshold}}
        (output / "pruning.pt").write_text("fake masks")
        (output / "baseline.json").write_text(json.dumps(summary))
        return summary

    # This fixture tests orchestration; test_pruning exercises real native pruning.
    state.prune = Mock(side_effect=prune)
    monkeypatch.setattr(runner, "save_baseline", state.prune)
    state.cuda = SimpleNamespace(
        is_available=Mock(return_value=True), is_bf16_supported=Mock(return_value=True),
        get_device_name=Mock(return_value="mock CUDA"),
        get_device_properties=Mock(return_value=SimpleNamespace(total_memory=1024)),
        reset_peak_memory_stats=Mock(), max_memory_allocated=Mock(return_value=512),
    )

    # install: Register an in-memory module with the supplied attributes.
    def install(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    state.torch = install(
        "torch", cuda=state.cuda, bfloat16="bf16", device=lambda name: name,
        manual_seed=Mock(), isfinite=lambda matrix: SimpleNamespace(all=lambda: matrix.finite),
        count_nonzero=lambda matrix: SimpleNamespace(item=lambda: matrix.nonzero),
    )
    install("huggingface_hub", snapshot_download=state.download)
    install("transformers", AutoTokenizer=SimpleNamespace(from_pretrained=state.tokenizer_loader))
    install("circuit_tracer", attribute=state.attribute, ReplacementModel=SimpleNamespace(
        from_pretrained_and_transcoders=state.load_model,
    ))
    install("circuit_tracer.utils")
    install("circuit_tracer.utils.hf_utils", load_transcoders=state.load_clt)
    install("circuit_tracer.utils.create_graph_files", create_graph_files=state.export)
    state.distribution = Mock(read_text=Mock(return_value=json.dumps({
        "vcs_info": {"commit_id": runner.TRACER_REVISION}
    })))
    monkeypatch.setattr(runner.importlib.metadata, "distribution", lambda _: state.distribution)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "mock-version")
    # Restore these after each test; run() otherwise sets persistent defaults.
    for variable in ("HF_HOME", "HF_HUB_DOWNLOAD_TIMEOUT", "HF_HUB_ETAG_TIMEOUT"):
        monkeypatch.delenv(variable, raising=False)
    state.events = events
    state.output = tmp_path / "run"
    state.record = {"prompt_prefix": "The capital of France is", "answer": "Paris"}
    return state


# read_manifest: Read the saved run receipt for assertions.
def read_manifest(stack):
    return json.loads((stack.output / "manifest.json").read_text())


# test_run_wires_pinned_snapshot_cpu_clt_nnsight_and_exports: Check run wires pinned snapshot cpu clt nnsight and exports.
def test_run_wires_pinned_snapshot_cpu_clt_nnsight_and_exports(stack):
    receipt = runner.run(stack.record, stack.output, max_features=8,
                         node_threshold=0.7, edge_threshold=0.9)
    assert stack.events == [
        "model_snapshot", "clt_cpu", "replacement_model", "attribute", "pruning", "viewer"
    ]
    stack.prune.assert_called_once_with(stack.graph, stack.output / "baseline", 0.7, 0.9)
    assert receipt["baseline"]["thresholds"] == {"node": 0.7, "edge": 0.9}
    assert (stack.output / "baseline/pruning.pt").is_file()
    assert stack.export.call_args.kwargs["node_threshold"] == 0.7
    assert stack.export.call_args.kwargs["edge_threshold"] == 0.9
    stack.download.assert_called_once_with(
        repo_id=runner.MODEL_ID, revision=runner.MODEL_REVISION,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"], max_workers=2,
    )
    stack.tokenizer_loader.assert_called_once_with(str(stack.snapshot), local_files_only=True)
    stack.tokenizer.assert_called_once_with(stack.record["prompt_prefix"], add_special_tokens=True)
    stack.load_clt.assert_called_once_with(
        runner.clt_config(), device="cpu", dtype="bf16", lazy_encoder=False, lazy_decoder=False,
    )
    stack.load_model.assert_called_once_with(
        model_name=str(stack.snapshot.resolve()), transcoders=stack.clt,
        backend="nnsight", device="cuda", dtype="bf16",
    )
    stack.attribute.assert_called_once_with(
        stack.tokens, stack.model, max_feature_nodes=8, batch_size=1,
        max_n_logits=2, desired_logit_prob=0.95, verbose=True,
    )
    stack.graph.to.assert_called_once_with("cpu")
    stack.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)
    assert (stack.output / "graph.pt").read_text() == "fake graph"
    assert (stack.output / "viewer/fake.json").is_file()
    assert read_manifest(stack) == receipt
    assert receipt["status"] == receipt["stage"] == "complete"
    assert receipt["prompt_token_ids"] == stack.tokens
    assert receipt["graph"]["selected_features"] == 2
    assert receipt["graph"]["adjacency_convention"] == "row=target,column=source"
    assert receipt["peak_gpu_allocated_bytes"] == 512


# test_feature_cap_rejected_before_output_or_download: Check feature cap rejected before output or download.
@pytest.mark.parametrize("cap", [0, 129])
def test_feature_cap_rejected_before_output_or_download(stack, cap):
    with pytest.raises(ValueError, match="1..128"):
        runner.run(stack.record, stack.output, cap)
    assert not stack.output.exists()
    stack.download.assert_not_called()


# test_missing_dependency_is_a_controlled_failure_with_receipt: Check missing dependency is a controlled failure with receipt.
def test_missing_dependency_is_a_controlled_failure_with_receipt(stack, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="uv sync --locked --extra graph"):
        runner.run(stack.record, stack.output)
    receipt = read_manifest(stack)
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "dependencies"
    assert receipt["error"]["type"] == "RuntimeError"
    stack.download.assert_not_called()


# test_unsupported_gpu_stops_before_download: Check unsupported gpu stops before download.
@pytest.mark.parametrize("cuda,bf16", [(False, True), (True, False)])
def test_unsupported_gpu_stops_before_download(stack, cuda, bf16):
    stack.cuda.is_available.return_value = cuda
    stack.cuda.is_bf16_supported.return_value = bf16
    with pytest.raises(RuntimeError, match="native BF16"):
        runner.run(stack.record, stack.output)
    assert read_manifest(stack)["status"] == "failed"
    stack.download.assert_not_called()


# test_changed_tracer_pin_stops_before_download: Check changed tracer pin stops before download.
def test_changed_tracer_pin_stops_before_download(stack):
    stack.distribution.read_text.return_value = json.dumps({"vcs_info": {"commit_id": "other"}})
    with pytest.raises(RuntimeError, match="differs from the audited pin"):
        runner.run(stack.record, stack.output)
    assert read_manifest(stack)["stage"] == "dependencies"
    stack.download.assert_not_called()


# test_token_limit_stops_before_clt_loading: Check token limit stops before clt loading.
def test_token_limit_stops_before_clt_loading(stack):
    stack.tokens = list(range(33))
    with pytest.raises(ValueError, match="at most 32 tokens"):
        runner.run(stack.record, stack.output)
    assert stack.events == ["model_snapshot"]
    receipt = read_manifest(stack)
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "model_download"
    stack.load_clt.assert_not_called()


# test_missing_clt_layer_stops_before_model_loading: Check missing clt layer stops before model loading.
def test_missing_clt_layer_stops_before_model_loading(stack):
    stack.clt.n_layers = 25
    with pytest.raises(ValueError, match="all 26 CLT layers"):
        runner.run(stack.record, stack.output)
    assert read_manifest(stack)["stage"] == "clt_download_and_load"
    stack.load_model.assert_not_called()


# test_invalid_graph_does_not_export: Check invalid graph does not export.
@pytest.mark.parametrize("change", ["rank", "square", "finite", "feature_cap"])
def test_invalid_graph_does_not_export(stack, change):
    if change == "rank":
        stack.matrix.ndim = 1
    elif change == "square":
        stack.matrix.shape = (4, 3)
    elif change == "finite":
        stack.matrix.finite = False
    else:
        stack.graph.selected_features.numel.return_value = 129
    with pytest.raises(ValueError, match="attribution matrix|feature cap"):
        runner.run(stack.record, stack.output)
    assert read_manifest(stack)["stage"] == "attribution"
    assert read_manifest(stack)["status"] == "failed"
    stack.graph.to_pt.assert_not_called()
    stack.export.assert_not_called()


# test_export_failure_keeps_graph_and_records_failed_stage: Check export failure keeps graph and records failed stage.
def test_export_failure_keeps_graph_and_records_failed_stage(stack):
    stack.export.side_effect = OSError("mock disk failure")
    with pytest.raises(OSError, match="mock disk failure"):
        runner.run(stack.record, stack.output)
    receipt = read_manifest(stack)
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "viewer_export"
    assert receipt["error"] == {"type": "OSError", "message": "mock disk failure"}
    assert receipt["elapsed_seconds"] >= 0
    assert (stack.output / "graph.pt").is_file()


# test_existing_output_is_preserved_without_any_download: Check existing output is preserved without any download.
def test_existing_output_is_preserved_without_any_download(stack):
    stack.output.mkdir()
    (stack.output / "manifest.json").write_text('{"status":"complete"}\n')
    (stack.output / "graph.pt").write_bytes(b"previous graph")
    before = {path.name: path.read_bytes() for path in stack.output.iterdir()}
    with pytest.raises(FileExistsError):
        runner.run(stack.record, stack.output)
    assert {path.name: path.read_bytes() for path in stack.output.iterdir()} == before
    stack.download.assert_not_called()


# test_pruning_thresholds_rejected_before_output_or_download: Check pruning thresholds rejected before output or download.
@pytest.mark.parametrize("node,edge", [(0, 0.98), (0.8, 1.1), (float("nan"), 0.98)])
def test_pruning_thresholds_rejected_before_output_or_download(stack, node, edge):
    with pytest.raises(ValueError):
        runner.run(stack.record, stack.output, node_threshold=node, edge_threshold=edge)
    assert not stack.output.exists()
    stack.download.assert_not_called()


# test_dry_run_rejects_invalid_pruning_threshold: Check dry run rejects invalid pruning threshold.
def test_dry_run_rejects_invalid_pruning_threshold(tmp_path):
    output = tmp_path / "absent"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--prompt", "France", "--dry-run", "--node-threshold", "0",
                     "--output", str(output)])
    assert exc.value.code == 1
    assert not output.exists()


# test_pruning_failure_preserves_graph_and_skips_viewer: Check pruning failure preserves graph and skips viewer.
def test_pruning_failure_preserves_graph_and_skips_viewer(stack):
    stack.prune.side_effect = ValueError("no output-connected influence")
    with pytest.raises(ValueError, match="no output-connected"):
        runner.run(stack.record, stack.output)
    receipt = read_manifest(stack)
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "baseline_pruning"
    assert (stack.output / "graph.pt").is_file()
    stack.export.assert_not_called()
