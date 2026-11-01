# prune_graph.py
# Export for doing attribution-only pruning (baseline approach) masks while also keeping the original graph's weights.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circuit_tracer.graph import Graph


# validate_thresholds: Require finite fractions in (0, 1]; smaller values prune more.
def validate_thresholds(node_threshold: float, edge_threshold: float) -> None:
    for name, value in (("node_threshold", node_threshold), ("edge_threshold", edge_threshold)):
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be finite and in (0, 1]")


def save_baseline(
    graph: Graph, output: Path, node_threshold: float = 0.8, edge_threshold: float = 0.98
) -> dict:
    validate_thresholds(node_threshold, edge_threshold)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    try:
        import torch
        from circuit_tracer.graph import compute_node_influence, prune_graph
    except ImportError as exc:
        raise RuntimeError("Install the locked graph extra before pruning a graph") from exc

    matrix = graph.adjacency_matrix
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not matrix.is_floating_point() or not torch.isfinite(matrix).all()):
        raise ValueError("Graph adjacency must be a finite square floating-point matrix")
    n_tokens, n_logits, n_features = (
        len(graph.input_tokens), len(graph.logit_targets), len(graph.selected_features)
    )
    expected = n_features + graph.cfg.n_layers * n_tokens + n_tokens + n_logits
    if (n_tokens < 1 or n_logits < 1 or graph.cfg.n_layers < 1
            or graph.input_tokens.ndim != 1 or graph.selected_features.ndim != 1
            or graph.n_pos != n_tokens or matrix.shape[0] != expected):
        raise ValueError("Graph dimensions do not match its feature/error/token/logit metadata")
    if (graph.active_features.ndim != 2 or graph.active_features.shape[1] != 3
            or (n_features and (graph.selected_features.min() < 0
                               or graph.selected_features.max() >= len(graph.active_features)))):
        raise ValueError("Selected features must index the graph's active feature records")
    probabilities = graph.logit_probabilities
    if (probabilities.shape != (n_logits,) or not torch.isfinite(probabilities).all()
            or not (probabilities > 0).all()):
        raise ValueError("Every logit target must have a finite positive weight")

    # Rank output influence using absolute, row-normalized edges.
    weights = torch.zeros(matrix.shape[0], device=matrix.device)
    weights[-n_logits:] = probabilities
    influence = compute_node_influence(matrix, weights)
    if not torch.isfinite(influence).all() or not influence.sum() > 0:
        raise ValueError("Graph has no finite positive influence on its output logits")
    native = prune_graph(graph, node_threshold, edge_threshold)
    nodes = native.node_mask
    # Keep only nonzero edges between retained nodes.
    edges = native.edge_mask & (matrix != 0) & nodes[:, None] & nodes[None, :]
    if not torch.isfinite(native.cumulative_scores).all() or not edges[-n_logits:].any():
        raise ValueError("Pruning retained no valid output-connected edges; raise the thresholds")

    summary = {
        "method": "circuit_tracer.graph.prune_graph",
        "thresholds": {"node": node_threshold, "edge": edge_threshold},
        "before": {"nodes": matrix.shape[0], "feature_nodes": n_features,
                   "edges": int(torch.count_nonzero(matrix).item())},
        "after": {"nodes": int(nodes.sum().item()),
                  "feature_nodes": int(nodes[:n_features].sum().item()),
                  "edges": int(edges.sum().item())},
        "artifacts": {"masks": "pruning.pt", "summary": "baseline.json"},
        "adjacency_convention": "row=target,column=source",
        "semantics": [
            "Attribution-only selection using absolute row-normalized influence.",
            "Nodes are selected first; edge influence is then recomputed on that subgraph.",
            "Native cleanup removes dangling feature/error nodes; tokens and logits stay.",
            "Threshold ties may retain extra nodes or edges; this is not exact-K selection.",
            "At 1.0, disconnected zero-influence edges need not be retained.",
            "Masks index the original graph; original signed edge weights are unchanged.",
            "This is not an intervention, a faithfulness measurement, or the proposed method.",
        ],
    }
    # Create after validation; refuse to overwrite an existing run.
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"node_mask": nodes.cpu(), "edge_mask": edges.cpu()}, output / "pruning.pt")
    (output / "baseline.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# main
def main() -> None:
    parser = argparse.ArgumentParser(description=(
        "Export the pinned circuit-tracer attribution-only pruning baseline.\n\n"
        "The original graph keeps its signed weights; this module saves selection masks,\n"
        "not a new attribution graph or an intervention result. Optional graph dependencies\n"
        "are imported only when pruning, so importing this module remains lightweight.\n"
        "Reuse a trusted local graph with: python prune_graph.py --graph graph.pt --output baseline\n"
    ))
    parser.add_argument("--graph", type=Path, required=True, help="Trusted local graph.pt file")
    parser.add_argument("--output", type=Path, required=True, help="Fresh baseline directory")
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    args = parser.parse_args()
    try:
        validate_thresholds(args.node_threshold, args.edge_threshold)
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
        from circuit_tracer.graph import Graph

        # .pt loading uses pickle, so the graph file must be trusted.
        graph = Graph.from_pt(str(args.graph), map_location="cpu")
        summary = save_baseline(graph, args.output, args.node_threshold, args.edge_threshold)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
