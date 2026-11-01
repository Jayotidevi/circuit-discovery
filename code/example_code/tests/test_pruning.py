# # test_pruning.py
# Offline native-pruning checks on synthetic graphs; no model weights are loaded.
#

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import prune_graph as baseline


# graph: A branching DAG with one negative edge and one disconnected feature.
@pytest.fixture
def graph():
    torch = pytest.importorskip("torch")
    upstream = pytest.importorskip("circuit_tracer.graph")
    from circuit_tracer.attribution.targets import LogitTarget
    from circuit_tracer.utils.tl_nnsight_mapping import UnifiedConfig

    config = UnifiedConfig(2, 4, 2, 2, 8, 10, "synthetic", "synthetic", "synthetic")
    matrix = torch.zeros(8, 8)
    # Four features, two error nodes, one token, then one output logit.
    for target, source, weight in (
        (0, 6, 1), (1, 6, 1), (2, 0, -2), (2, 1, 1),
        (7, 2, 9), (7, 4, 1), (3, 6, 4),
    ):
        matrix[target, source] = weight
    return upstream.Graph(
        input_string="Synthetic prompt", input_tokens=torch.tensor([1]),
        active_features=torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]),
        adjacency_matrix=matrix, cfg=config, selected_features=torch.arange(4),
        activation_values=torch.ones(4),
        logit_targets=[LogitTarget(token_str="answer", vocab_idx=2)],
        logit_probabilities=torch.tensor([0.7]), scan_name="synthetic-pruning-test",
    )


# read_masks: Load saved pruning masks on CPU.
def read_masks(output):
    import torch

    return torch.load(output / "pruning.pt", weights_only=True, map_location="cpu")


# test_default_masks_match_native_and_preserve_signed_graph: Check default masks match native and preserve signed graph.
def test_default_masks_match_native_and_preserve_signed_graph(graph, tmp_path):
    import torch
    from circuit_tracer.graph import prune_graph

    original = graph.adjacency_matrix.clone()
    native = prune_graph(graph, 0.8, 0.98)
    expected_edges = (native.edge_mask & (original != 0)
                      & native.node_mask[:, None] & native.node_mask[None, :])
    output = tmp_path / "baseline"
    summary = baseline.save_baseline(graph, output)
    masks = read_masks(output)
    torch.testing.assert_close(masks["node_mask"], native.node_mask)
    torch.testing.assert_close(masks["edge_mask"], expected_edges)
    torch.testing.assert_close(graph.adjacency_matrix, original)
    assert masks["edge_mask"][2, 0] and original[2, 0] == -2
    assert masks["edge_mask"].dtype == masks["node_mask"].dtype == torch.bool
    assert summary["before"] == {"nodes": 8, "feature_nodes": 4, "edges": 7}
    assert summary["after"] == {"nodes": 4, "feature_nodes": 2, "edges": 3}
    assert summary["thresholds"] == {"node": 0.8, "edge": 0.98}
    assert json.loads((output / "baseline.json").read_text()) == summary
    assert set(path.name for path in output.iterdir()) == {"baseline.json", "pruning.pt"}


# test_stricter_node_threshold_reduces_native_retention: Check stricter node threshold reduces native retention.
def test_stricter_node_threshold_reduces_native_retention(graph, tmp_path):
    full = baseline.save_baseline(graph, tmp_path / "full", 1.0, 1.0)
    smaller = baseline.save_baseline(graph, tmp_path / "smaller", 0.8, 1.0)
    assert smaller["after"]["nodes"] < full["after"]["nodes"]
    assert smaller["after"]["edges"] < full["after"]["edges"]


# test_stricter_edge_threshold_reduces_retention: Check stricter edge threshold reduces retention.
def test_stricter_edge_threshold_reduces_retention(graph, tmp_path):
    # A strong direct error-to-logit edge survives while a weaker branch is removed.
    graph.adjacency_matrix.zero_()
    graph.adjacency_matrix[7, 4] = 9
    graph.adjacency_matrix[7, 5] = 1
    full = baseline.save_baseline(graph, tmp_path / "full", 1.0, 1.0)
    smaller = baseline.save_baseline(graph, tmp_path / "smaller", 1.0, 0.8)
    assert full["after"]["edges"] == 2
    assert smaller["after"]["edges"] == 1
    assert smaller["after"]["nodes"] == 3  # Error, forced token, forced logit.


# test_full_threshold_keeps_reachable_edges_not_disconnected_edges: Check full threshold keeps reachable edges not disconnected edges.
def test_full_threshold_keeps_reachable_edges_not_disconnected_edges(graph, tmp_path):
    summary = baseline.save_baseline(graph, tmp_path / "full", 1.0, 1.0)
    masks = read_masks(tmp_path / "full")
    assert summary["after"] == {"nodes": 6, "feature_nodes": 3, "edges": 6}
    assert not masks["node_mask"][3] and not masks["edge_mask"][3, 6]
    assert not (masks["edge_mask"] & (graph.adjacency_matrix == 0)).any()


# test_ties_follow_native_threshold_not_exact_k: Check ties follow native threshold not exact k.
def test_ties_follow_native_threshold_not_exact_k(graph, tmp_path):
    graph.adjacency_matrix[2, 0] = -1
    graph.adjacency_matrix[7, 4] = 0
    summary = baseline.save_baseline(graph, tmp_path / "ties", 0.7, 0.8)
    assert summary["after"] == {"nodes": 5, "feature_nodes": 3, "edges": 5}


# test_invalid_thresholds_leave_no_output: Check invalid thresholds leave no output.
@pytest.mark.parametrize("value", [0, -0.1, 1.01, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("which", ["node", "edge"])
def test_invalid_thresholds_leave_no_output(graph, tmp_path, value, which):
    thresholds = (value, 0.98) if which == "node" else (0.8, value)
    with pytest.raises(ValueError, match="finite and in"):
        baseline.save_baseline(graph, tmp_path / "bad", *thresholds)
    assert not (tmp_path / "bad").exists()


# test_invalid_graphs_leave_no_output: Check invalid graphs leave no output.
@pytest.mark.parametrize("problem", [
    "nonsquare", "nan", "inf", "integer", "metadata", "position", "feature_index",
    "no_logits", "weight_shape", "zero_weight", "negative_weight", "nan_weight",
    "zero_matrix", "disconnected_only", "stage_two_zero",
])
def test_invalid_graphs_leave_no_output(graph, tmp_path, problem):
    import torch

    if problem == "nonsquare":
        graph.adjacency_matrix = torch.zeros(8, 7)
    elif problem in {"nan", "inf"}:
        graph.adjacency_matrix[0, 6] = float(problem)
    elif problem == "integer":
        graph.adjacency_matrix = graph.adjacency_matrix.long()
    elif problem == "metadata":
        graph.cfg.n_layers = 3
    elif problem == "position":
        graph.n_pos = 2
    elif problem == "feature_index":
        graph.selected_features[0] = 99
    elif problem == "no_logits":
        graph.logit_targets = []
    elif problem == "weight_shape":
        graph.logit_probabilities = torch.ones(2)
    elif problem.endswith("_weight"):
        graph.logit_probabilities[0] = {
            "zero_weight": 0, "negative_weight": -1, "nan_weight": float("nan")
        }[problem]
    else:
        graph.adjacency_matrix.zero_()
        if problem == "disconnected_only":
            graph.adjacency_matrix[0, 6] = 1
        elif problem == "stage_two_zero":
            # Node 0 collects both branches' influence. A low cutoff drops both
            # logit inputs, leaving a disconnected token->0 edge upstream may keep.
            for target, source in ((0, 6), (1, 0), (2, 0), (7, 1), (7, 2)):
                graph.adjacency_matrix[target, source] = 1
    with pytest.raises(ValueError):
        baseline.save_baseline(graph, tmp_path / "bad", 0.1 if problem == "stage_two_zero" else 0.8)
    assert not (tmp_path / "bad").exists()


# test_existing_output_is_preserved: Check existing output is preserved.
def test_existing_output_is_preserved(graph, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "baseline.json"
    sentinel.write_text("keep me")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        baseline.save_baseline(graph, output)
    assert sentinel.read_text() == "keep me"
    assert not (output / "pruning.pt").exists()


# test_cli_roundtrip_uses_saved_graph_offline: Check cli roundtrip uses saved graph offline.
def test_cli_roundtrip_uses_saved_graph_offline(graph, tmp_path):
    graph_path = tmp_path / "graph.pt"
    graph.to_pt(str(graph_path))
    saved_bytes = graph_path.read_bytes()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, str(Path(baseline.__file__)), "--graph", str(graph_path),
         "--output", str(tmp_path / "cli")],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["after"]["edges"] == 3
    assert graph_path.read_bytes() == saved_bytes
    assert read_masks(tmp_path / "cli")["edge_mask"].device.type == "cpu"


# test_import_and_threshold_validation_require_only_stdlib: Check import and threshold validation require only stdlib.
def test_import_and_threshold_validation_require_only_stdlib():
    script = (
        "import runpy,sys; "
        f"m=runpy.run_path({str(Path(baseline.__file__))!r}); "
        "m['validate_thresholds'](.8,.98); "
        "assert 'torch' not in sys.modules and 'circuit_tracer' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
