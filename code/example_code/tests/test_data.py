# # test_data.py
# Focused offline checks for the simple factual-prompt preparation script.
#

import hashlib
import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest

import prepare_data as data

ROOT = data.ROOT
REVISION = "a" * 40


# fixture: Write one small pinned source so ordinary tests need neither cache nor network.
def fixture(root: Path, *, cached: bool = True, value: dict | None = None) -> tuple[dict, bytes]:
    value = (
        value
        if value is not None
        else {
            "prompt_templates": ["{} is part of the country of"],
            "samples": [
                {"subject": "Rio de Janeiro", "object": "Brazil"},
                {"subject": "Buenos Aires", "object": "Argentina"},
                {"subject": "Mexico City", "object": "Mexico"},
                {"subject": "Tirana", "object": "Iran"},  # Substrings are not word overlap.
            ],
        }
    )
    content = json.dumps(value).encode()
    path = "dataset/relations_data/semantic/city_in_country.json"
    entry = {
        "path": path,
        "category": "semantic",
        "relation": "city_in_country",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "url": f"https://raw.githubusercontent.com/adhakimi/time-course-mechinterp/{REVISION}/{path}",
    }
    lock = {
        "schema_version": 1,
        "repository": data.REPOSITORY,
        "revision": REVISION,
        "files": [entry],
    }
    (root / "datasets").mkdir(parents=True)
    (root / "datasets/time_course.lock.json").write_text(json.dumps(lock))
    if cached:
        raw = root / "datasets/raw/time-course-mechinterp/semantic/city_in_country.json"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(content)
    return entry, content


# test_prepare_preserves_prefix_answer_provenance_and_is_deterministic: Check prepare preserves prefix answer provenance and is deterministic.
def test_prepare_preserves_prefix_answer_provenance_and_is_deterministic(tmp_path, monkeypatch):
    entry, _ = fixture(tmp_path)
    monkeypatch.setattr(
        data, "urlopen", lambda *args, **kwargs: pytest.fail("valid cache should be offline")
    )
    result = data.prepare_data(tmp_path)
    assert result["source_files"] == 1
    assert (result["candidates"], result["excluded"], result["prompts"]) == (4, 1, 3)
    assert result["excluded_subjects"] == ["Mexico City"]
    output = Path(result["output"])
    before = output.read_bytes()
    rows = [json.loads(line) for line in before.splitlines()]
    assert len({row["id"] for row in rows}) == 3
    assert rows[0]["prompt_prefix"] == "Rio de Janeiro is part of the country of"
    assert rows[0]["answer"] == "Brazil"
    assert rows[0]["source"] == {
        "repository": data.REPOSITORY,
        "revision": REVISION,
        "path": entry["path"],
        "sha256": entry["sha256"],
        "sample_index": 0,
        "template_index": 0,
    }
    assert "Brazil" not in rows[0]["prompt_prefix"]
    assert rows[-1]["subject"] == "Tirana"
    assert not any("split" in row or "donor" in row for row in rows)
    assert data.prepare_data(tmp_path) == result
    assert output.read_bytes() == before


# test_corrupt_cached_file_fails_without_overwriting_or_downloading: Check corrupt cached file fails without overwriting or downloading.
def test_corrupt_cached_file_fails_without_overwriting_or_downloading(tmp_path, monkeypatch):
    fixture(tmp_path)
    raw = tmp_path / "datasets/raw/time-course-mechinterp/semantic/city_in_country.json"
    raw.write_bytes(b"corrupt")
    monkeypatch.setattr(
        data, "urlopen", lambda *args, **kwargs: pytest.fail("must preserve corrupt cache")
    )
    with pytest.raises(ValueError, match="Checksum/size mismatch"):
        data.prepare_data(tmp_path)
    assert raw.read_bytes() == b"corrupt"
    assert not (tmp_path / "datasets/prepared").exists()


# test_missing_cache_downloads_exact_pinned_url_with_timeout: Check missing cache downloads exact pinned url with timeout.
def test_missing_cache_downloads_exact_pinned_url_with_timeout(tmp_path, monkeypatch):
    entry, content = fixture(tmp_path, cached=False)
    calls = []

    # download: Return pinned fixture bytes through a fake download response.
    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(content)

    monkeypatch.setattr(data, "urlopen", download)
    assert data.prepare_data(tmp_path)["prompts"] == 3
    assert calls == [(entry["url"], 30)]
    assert (
        tmp_path / "datasets/raw/time-course-mechinterp/semantic/city_in_country.json"
    ).read_bytes() == content


# test_network_failure_reports_source_and_leaves_no_prepared_output: Check network failure reports source and leaves no prepared output.
@pytest.mark.parametrize("error", [TimeoutError("timed out"), URLError("connection failed")])
def test_network_failure_reports_source_and_leaves_no_prepared_output(tmp_path, monkeypatch, error):
    fixture(tmp_path, cached=False)

    # fail: Raise the configured failure to exercise error handling.
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(data, "urlopen", fail)
    with pytest.raises(ValueError, match="Download failed for.*city_in_country.json"):
        data.prepare_data(tmp_path)
    assert not (tmp_path / "datasets/prepared").exists()


# test_bad_download_is_not_cached: Check bad download is not cached.
def test_bad_download_is_not_cached(tmp_path, monkeypatch):
    fixture(tmp_path, cached=False)
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad data"))
    with pytest.raises(ValueError, match="Checksum/size mismatch"):
        data.prepare_data(tmp_path)
    assert not (tmp_path / "datasets/raw").exists()


# test_malformed_source_schema_is_explicit: Check malformed source schema is explicit.
@pytest.mark.parametrize(
    "value",
    [
        {"samples": [], "prompt_templates": ["{} is in"]},
        {"samples": [{"subject": "Rio", "object": "Brazil"}], "prompt_templates": ["{answer}"]},
        {"samples": [{"subject": "Rio", "object": "Brazil"}], "prompt_templates": ["{} {}"]},
        {"samples": [{"subject": "Rio"}], "prompt_templates": ["{} is in"]},
    ],
)
def test_malformed_source_schema_is_explicit(tmp_path, value):
    fixture(tmp_path, value=value)
    with pytest.raises(ValueError, match="samples|placeholder|subject/object"):
        data.prepare_data(tmp_path)


# test_repository_lock_contains_both_pinned_categories: Check repository lock contains both pinned categories.
def test_repository_lock_contains_both_pinned_categories():
    lock = json.loads((ROOT / "datasets/time_course.lock.json").read_text())
    assert lock["revision"] == "0f018bbc8bd766cdf74930c445f0c9c4df9ac7bd"
    assert len(lock["files"]) == 37
    assert sum(entry["category"] == "semantic" for entry in lock["files"]) == 10
    assert sum(entry["category"] == "factual" for entry in lock["files"]) == 27
