# # test_notebook.py
# Run notebook orchestration offline; no Colab, model weights, or GPU required.
#

import ast
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import uuid
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "code/example_code"
NOTEBOOK = EXAMPLE / "gemma_scope2_baseline.ipynb"


# cells: Read the notebook cells without executing them.
def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


# execute: Run one notebook cell in the supplied shared namespace.
def execute(cell_id, namespace):
    cell = next(cell for cell in cells() if cell["id"] == cell_id)
    exec(compile("".join(cell["source"]), f"{NOTEBOOK}:{cell_id}", "exec"), namespace)


# project: Create a temporary checkout and notebook namespace for offline tests.
@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    example = root / "code/example_code"
    example.mkdir(parents=True)
    for name in ("prepare_data.py", "generate_graph.py", "prune_graph.py"):
        shutil.copyfile(EXAMPLE / name, example / name)
    for name in ("uv.lock", "pyproject.toml"):
        (root / name).write_text("# Local root marker for this test\n")
    output = example / "output"
    output.mkdir()
    return {"ROOT": root, "OUT": output, "Path": Path, "json": json, "os": os,
            "shutil": shutil, "subprocess": subprocess, "sys": sys, "uuid": uuid,
            "PY": [sys.executable], "ROW": 7, "IN_COLAB": False}


# mock_colab: Install fake Colab services for notebook tests.
def mock_colab(monkeypatch, **services):
    google, colab = ModuleType("google"), ModuleType("google.colab")
    colab.__dict__.update(services)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


# test_notebook_is_unexecuted_python_with_six_ordered_steps: Check notebook is unexecuted python with six ordered steps.
def test_notebook_is_unexecuted_python_with_six_ordered_steps():
    book = json.loads(NOTEBOOK.read_text())
    assert book["nbformat"] == 4
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert [cell["id"] for cell in code] == [
        "project", "environment", "data", "access", "graph", "download"
    ]
    for cell in code:
        assert cell["execution_count"] is None and cell["outputs"] == []
        ast.parse("".join(cell["source"]))


# test_project_cell_finds_ancestor_checkout: Check project cell finds ancestor checkout.
def test_project_cell_finds_ancestor_checkout(project, monkeypatch):
    monkeypatch.chdir(project["OUT"])
    monkeypatch.setenv("HF_HOME", "restored-after-test")
    namespace = {}
    execute("project", namespace)
    assert namespace["ROOT"] == project["ROOT"]
    assert namespace["OUT"] == project["OUT"]
    assert Path.cwd() == project["ROOT"]
    assert Path(os.environ["HF_HOME"]) == project["ROOT"] / ".cache/huggingface"


# test_project_rejects_unsafe_or_incomplete_zip_before_extraction: Check project rejects unsafe or incomplete zip before extraction.
@pytest.mark.parametrize("members", [
    ["../escape.py"], ["/absolute.py"], ["nested/uv.lock"],
    # A previously valid bundle is incomplete without the new pruning helper.
    ["pyproject.toml", "uv.lock", ".python-version", "datasets/time_course.lock.json",
     "code/example_code/prepare_data.py", "code/example_code/generate_graph.py"],
])
def test_project_rejects_unsafe_or_incomplete_zip_before_extraction(tmp_path, monkeypatch, members):
    monkeypatch.chdir(tmp_path)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for member in members:
            archive.writestr(member, "must not be extracted")
    files = SimpleNamespace(upload=lambda: {"cs795-colab.zip": payload.getvalue()})
    mock_colab(monkeypatch, files=files)
    # Even a regression cannot write into /content during this rejection test.
    extract = Mock(side_effect=AssertionError("Archive must be rejected before extraction"))
    monkeypatch.setattr(zipfile.ZipFile, "extractall", extract)
    with pytest.raises(ValueError, match="unsafe path|repository-relative"):
        execute("project", {})
    extract.assert_not_called()


# test_local_project_cell_reports_missing_checkout: Check local project cell reports missing checkout.
def test_local_project_cell_reports_missing_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(RuntimeError, match="inside the project repository"):
        execute("project", {})


# test_data_cell_prepares_cached_sources_and_copies_separate_answers: Check data cell prepares cached sources and copies separate answers.
@pytest.mark.skipif(
    not (ROOT / "datasets/raw/time-course-mechinterp").is_dir(),
    reason="Run prepare_data.py once to enable the cached-dataset integration check.",
)
def test_data_cell_prepares_cached_sources_and_copies_separate_answers(project, monkeypatch):
    target = project["ROOT"] / "datasets"
    target.mkdir()
    shutil.copyfile(ROOT / "datasets/time_course.lock.json", target / "time_course.lock.json")
    shutil.copytree(ROOT / "datasets/raw", target / "raw")
    assert len(list((target / "raw").rglob("*.json"))) == 37
    # A different notebook cwd checks the cell's explicit subprocess project cwd.
    monkeypatch.chdir(project["OUT"])
    execute("data", project)
    copied = project["OUT"] / "prompts.jsonl"
    assert copied.read_bytes() == project["prepared"].read_bytes()
    rows = [json.loads(line) for line in copied.read_text().splitlines()]
    assert len(rows) == 31 and project["ROW"] == 0
    for row in rows:
        assert row["answer"] and row["answer"].casefold() not in row["prompt_prefix"].casefold()
        assert row["source"]["revision"] and row["source"]["sha256"]


# test_hardware_failure_stops_before_secret_lookup: Check hardware failure stops before secret lookup.
def test_hardware_failure_stops_before_secret_lookup(project, monkeypatch):
    secrets = SimpleNamespace(get=Mock(side_effect=AssertionError("Must not request a token")))
    mock_colab(monkeypatch, userdata=secrets)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    project["IN_COLAB"] = True
    project["subprocess"] = SimpleNamespace(run=Mock(side_effect=subprocess.CalledProcessError(1, "probe")))
    with pytest.raises(subprocess.CalledProcessError):
        execute("access", project)
    secrets.get.assert_not_called()


# test_graph_cell_uses_selected_row_and_fresh_directories: Check graph cell uses selected row and fresh directories.
def test_graph_cell_uses_selected_row_and_fresh_directories(project, capsys):
    runs = []

    # run: Capture the notebook command and simulate a successful graph run.
    def run(argv, *, check, cwd, timeout):
        assert argv[:2] == [sys.executable, "code/example_code/generate_graph.py"]
        assert argv[argv.index("--row") + 1] == "7" and check is True
        assert float(argv[argv.index("--node-threshold") + 1]) == project["NODE_THRESHOLD"] == 0.8
        assert float(argv[argv.index("--edge-threshold") + 1]) == project["EDGE_THRESHOLD"] == 0.98
        assert cwd == project["ROOT"] and timeout > 0
        destination = Path(argv[argv.index("--output") + 1])
        destination.mkdir()  # A repeated directory would fail, preserving prior output.
        (destination / "graph.pt").write_bytes(b"mock graph")
        (destination / "manifest.json").write_text(json.dumps({
            "status": "complete", "graph": {"nodes": 3},
            "baseline": {"node_threshold": 0.8, "edge_threshold": 0.98, "retained_nodes": 2},
        }))
        runs.append(destination)

    project["subprocess"] = SimpleNamespace(run=run)
    execute("graph", project)
    execute("graph", project)
    assert len(set(runs)) == 2 and all(path.parent == project["OUT"] for path in runs)
    assert all((path / "graph.pt").read_bytes() == b"mock graph" for path in runs)
    printed = capsys.readouterr().out
    assert project["RUN_DIR"] == runs[-1] and '"nodes": 3' in printed
    assert '"baseline"' in printed and '"retained_nodes": 2' in printed


# test_graph_failure_displays_and_preserves_saved_receipt: Check graph failure displays and preserves saved receipt.
def test_graph_failure_displays_and_preserves_saved_receipt(project, capsys):
    receipt = {"status": "failed", "stage": "model_download", "error": {"message": "mock denied"}}

    # fail: Raise the configured failure to exercise error handling.
    def fail(argv, **kwargs):
        destination = Path(argv[argv.index("--output") + 1])
        destination.mkdir()
        (destination / "manifest.json").write_text(json.dumps(receipt))
        raise subprocess.CalledProcessError(1, argv)

    project["subprocess"] = SimpleNamespace(run=fail)
    with pytest.raises(subprocess.CalledProcessError):
        execute("graph", project)
    assert json.loads((project["RUN_DIR"] / "manifest.json").read_text()) == receipt
    assert "mock denied" in capsys.readouterr().out


# test_download_contains_only_current_run_and_reports_local_filename: Check download contains only current run and reports local filename.
def test_download_contains_only_current_run_and_reports_local_filename(project, capsys):
    run = project["OUT"] / "graph-current"
    run.mkdir()
    (run / "manifest.json").write_text('{"status":"failed"}')
    (run / "graph.pt").write_bytes(b"mock graph")
    (run / "baseline").mkdir()
    (run / "baseline/baseline.json").write_text('{"retained_nodes":2}')
    (run / "baseline/pruning.pt").write_bytes(b"mock masks")
    # These siblings must never enter the downloadable run archive.
    (project["ROOT"] / ".cache").mkdir()
    (project["ROOT"] / ".cache/weights").write_text("large weights")
    (project["ROOT"] / ".env").write_text("HF_TOKEN=not-a-real-token")
    (project["OUT"] / "prompts.jsonl").write_text("separate dataset")
    project["RUN_DIR"] = run
    execute("download", project)
    archive = Path(project["archive"])
    assert archive == run.with_suffix(".zip")
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {
            "manifest.json", "graph.pt", "baseline/", "baseline/baseline.json", "baseline/pruning.pt"
        }
    assert str(archive) in capsys.readouterr().out
