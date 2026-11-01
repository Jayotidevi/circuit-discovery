# # test_upstream_integration.py
# Offline compatibility check using synthetic weights, not real Gemma acceptance.
#

import json
from pathlib import Path

import pytest

from prune_graph import save_baseline


# test_synthetic_scope2_clt_to_viewer_offline: Check synthetic scope2 clt to viewer offline.
def test_synthetic_scope2_clt_to_viewer_offline(tmp_path, monkeypatch):
    # Set these before optional imports; all model/tokenizer assets are created here.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    safetensors = pytest.importorskip("safetensors.torch")
    pytest.importorskip("nnsight")
    tracer = pytest.importorskip("circuit_tracer")
    clt_module = pytest.importorskip("circuit_tracer.transcoder.cross_layer_transcoder")
    export_module = pytest.importorskip("circuit_tracer.utils.create_graph_files")
    graph_module = pytest.importorskip("circuit_tracer.graph")
    hub_constants = pytest.importorskip("huggingface_hub.constants")
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    # The upstream exporter otherwise selects CUDA whenever one is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    generator = torch.Generator().manual_seed(0)
    n_layers, d_model, d_feature = 2, 16, 4
    paths = {}
    for layer in range(n_layers):
        path = tmp_path / f"params_layer_{layer}.safetensors"
        # These random tensors mimic the official raw Scope 2 format only.
        safetensors.save_file(
            {
                "w_enc": torch.randn(d_model, d_feature, generator=generator) * 0.1,
                "w_dec": torch.randn(d_feature, n_layers, d_model, generator=generator) * 0.1,
                "b_enc": torch.ones(d_feature) * 0.1,
                "b_dec": torch.zeros(d_model),
                "threshold": torch.zeros(d_feature),
                "affine_skip_connection": torch.eye(d_model) * 0.05,
            },
            str(path),
        )
        paths[layer] = str(path)

    # Commit 255d989 uses range(max(paths.keys())). The duplicate sentinel ensures
    # that every real layer is read once; it does not add another network layer.
    paths[n_layers] = paths[n_layers - 1]
    loader_kwargs = {
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "feature_input_hook": "mlp.hook_in",
        "feature_output_hook": "hook_mlp_out",
        "scan_name": "synthetic-test",
        "lazy_encoder": False,
        "lazy_decoder": False,
    }
    clt = clt_module.load_gemma_scope_2_clt(paths, **loader_kwargs)
    assert clt.n_layers == n_layers
    assert clt.W_enc.shape == (n_layers, d_feature, d_model)
    assert [tuple(weight.shape) for weight in clt.W_dec] == [
        (d_feature, 2, d_model), (d_feature, 1, d_model)
    ]
    with pytest.raises(RuntimeError, match="size mismatch for W_dec"):
        clt_module.load_gemma_scope_2_clt(
            {layer: paths[layer] for layer in range(n_layers)}, **loader_kwargs
        )

    # The raw loader leaves skip weights at their stored dtype. The .to() operation
    # used by the NNsight constructor is required before BF16 skip computation.
    bf16_clt = clt_module.load_gemma_scope_2_clt(
        paths, **(loader_kwargs | {"dtype": torch.bfloat16})
    )
    assert bf16_clt.W_enc.dtype == torch.bfloat16
    assert bf16_clt.W_skip.dtype == torch.float32
    bf16_clt.to(torch.device("cpu"), torch.bfloat16)
    assert bf16_clt.W_skip.dtype == torch.bfloat16
    skip = bf16_clt.compute_skip(0, torch.ones(3, d_model, dtype=torch.bfloat16))
    assert skip.dtype == torch.bfloat16
    assert torch.isfinite(skip).all()

    model_path = tmp_path / "model"
    model_path.mkdir()
    vocabulary = {
        "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
        "The": 4, "capital": 5, "of": 6, "France": 7, "is": 8, "Paris": 9,
    }
    raw_tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocabulary, unk_token="<unk>")
    )
    raw_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    raw_tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 2)]
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer, bos_token="<bos>", eos_token="<eos>",
        pad_token="<pad>", unk_token="<unk>",
    )
    tokenizer.save_pretrained(model_path)
    config = transformers.Gemma3TextConfig(
        vocab_size=len(vocabulary), hidden_size=d_model, intermediate_size=32,
        num_hidden_layers=n_layers, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=128, sliding_window=32,
        sliding_window_pattern=2,
    )
    # fork_rng keeps the fixture deterministic without changing other tests' RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        tiny_model = transformers.Gemma3ForCausalLM(config).eval()
    tiny_model.save_pretrained(model_path)
    replacement = tracer.ReplacementModel.from_pretrained_and_transcoders(
        model_name=str(model_path), transcoders=clt, backend="nnsight",
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert Path(replacement.cfg.tokenizer_name).resolve() == model_path.resolve()
    assert replacement.transcoders.n_layers == n_layers
    assert replacement.transcoders.W_skip.dtype == torch.float32

    input_ids = tokenizer("The capital of France is")["input_ids"]
    assert input_ids == [2, 4, 5, 6, 7, 8]
    graph = tracer.attribute(
        input_ids, replacement, max_feature_nodes=4, max_n_logits=2,
        batch_size=1, verbose=False,
    )
    selected = graph.selected_features.numel()
    logits = graph.logit_token_ids.numel()
    assert 0 < selected <= 4
    assert 0 < logits <= 2
    assert graph.input_tokens.tolist() == input_ids
    assert torch.isfinite(graph.adjacency_matrix).all()
    expected_nodes = selected + n_layers * len(input_ids) + len(input_ids) + logits
    assert graph.adjacency_matrix.shape == (expected_nodes, expected_nodes)
    assert torch.count_nonzero(graph.adjacency_matrix) > 0

    graph.to("cpu")
    graph_path = tmp_path / "graph.pt"
    graph.to_pt(str(graph_path))
    restored = graph_module.Graph.from_pt(str(graph_path), map_location="cpu")
    torch.testing.assert_close(restored.adjacency_matrix, graph.adjacency_matrix)
    # Exercise the baseline on a real tracer graph, including its serialization.
    original = restored.adjacency_matrix.clone()
    summary = save_baseline(restored, tmp_path / "baseline")
    masks = torch.load(tmp_path / "baseline/pruning.pt", weights_only=True)
    native = graph_module.prune_graph(restored, node_threshold=0.8, edge_threshold=0.98)
    expected_edges = (native.edge_mask & original.ne(0)
                      & native.node_mask[:, None] & native.node_mask[None, :])
    assert torch.equal(masks["node_mask"], native.node_mask)
    assert torch.equal(masks["edge_mask"], expected_edges)
    assert summary["after"]["edges"] == expected_edges.count_nonzero().item()
    torch.testing.assert_close(restored.adjacency_matrix, original)
    viewer_path = tmp_path / "viewer"
    export_module.create_graph_files(
        restored, slug="tiny-smoke", output_path=str(viewer_path),
        node_threshold=0.8, edge_threshold=0.98,
    )
    exported = json.loads((viewer_path / "tiny-smoke.json").read_text())
    assert exported["metadata"]["slug"] == "tiny-smoke"
    assert exported["metadata"]["scan"] == "synthetic-test"
    assert exported["metadata"]["prompt_tokens"] == [
        "<bos>", "The", "capital", "of", "France", "is"
    ]
    assert exported["nodes"] and exported["links"]
    assert (viewer_path / "graph-metadata.json").is_file()
