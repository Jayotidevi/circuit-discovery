# # prepare_data.py
# Prepare a small factual-prompt set for the Scope 2 baseline, using only stdlib.
#

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import string
import sys
from typing import Any
import unicodedata
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "https://github.com/adhakimi/time-course-mechinterp"
RELATIONS = {"city_in_country", "country_capital_city"}


# _source_files: Read the lock, download missing files, and verify every cached byte.
def _source_files(root: Path) -> tuple[dict[str, Any], list[tuple[dict, dict]]]:
    lock = json.loads((root / "datasets/time_course.lock.json").read_text())
    revision = lock.get("revision", "")
    if lock.get("repository") != REPOSITORY or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset lock must specify the upstream repository and a full commit SHA")
    if (
        lock.get("schema_version") != 1
        or not isinstance(lock.get("files"), list)
        or not lock["files"]
    ):
        raise ValueError("Dataset lock must contain schema_version 1 and a nonempty files list")
    sources, seen = [], set()
    for entry in sorted(lock["files"], key=lambda item: item["path"]):
        path = entry["path"]
        match = re.fullmatch(r"dataset/relations_data/(semantic|factual)/([a-z0-9_]+)\.json", path)
        if not match or path in seen or match.groups() != (entry["category"], entry["relation"]):
            raise ValueError(f"Invalid or repeated dataset path: {path}")
        seen.add(path)
        url = f"https://raw.githubusercontent.com/adhakimi/time-course-mechinterp/{revision}/{path}"
        if (
            entry["url"] != url
            or type(entry["size"]) is not int
            or not 0 < entry["size"] < 5_000_000
        ):
            raise ValueError(f"Invalid pinned URL or byte count: {path}")
        local = (
            root
            / "datasets/raw/time-course-mechinterp"
            / entry["category"]
            / (entry["relation"] + ".json")
        )
        if local.exists():
            content = local.read_bytes()
        else:
            # Bound both network wait and bytes read; do not import upstream code.
            try:
                with urlopen(url, timeout=30) as response:
                    content = response.read(entry["size"] + 1)
            except (OSError, URLError) as exc:
                raise ValueError(f"Download failed for {path}: {exc}") from exc
        if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            # Preserve corrupt caches for inspection instead of silently replacing them.
            raise ValueError(f"Checksum/size mismatch: {path}")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON relation object: {path}")
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            temporary = local.with_suffix(".json.part")
            temporary.write_bytes(content)
            temporary.replace(local)
        sources.append((entry, value))
    return lock, sources


# _words: Normalize text into words for answer-overlap checks.
def _words(text: str) -> list[str]:
    # Whole-word matching catches Mexico in Mexico City, without Iran in Tirana.
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


# prepare_data: Verify all sources and write deterministic prompts with source provenance.
def prepare_data(root: Path = ROOT) -> dict[str, Any]:
    # Overlap checks do not establish model knowledge or token eligibility.
    lock, sources = _source_files(root)
    rows, excluded = [], []
    candidates = 0
    for entry, value in sources:
        if entry["category"] != "semantic" or entry["relation"] not in RELATIONS:
            continue
        templates, samples = value.get("prompt_templates"), value.get("samples")
        if (
            not isinstance(templates, list)
            or not templates
            or not isinstance(samples, list)
            or not samples
        ):
            raise ValueError(f"Missing prompt_templates or samples: {entry['path']}")
        for template_index, template in enumerate(templates):
            if not isinstance(template, str):
                raise ValueError(f"Template must be a string: {entry['path']}")
            fields = [
                part[1:] for part in string.Formatter().parse(template) if part[1] is not None
            ]
            if fields not in [[("", "", None)], [("0", "", None)]]:
                raise ValueError(f"Expected one plain subject placeholder: {entry['path']}")
            for sample_index, sample in enumerate(samples):
                if not isinstance(sample, dict) or any(
                    not isinstance(sample.get(k), str) or not sample[k].strip()
                    for k in ("subject", "object")
                ):
                    raise ValueError(
                        f"Invalid subject/object at {entry['path']} sample {sample_index}"
                    )
                subject, answer = sample["subject"], sample["object"]
                prompt = template.format(subject)  # Never append the target answer.
                prompt_words, answer_words = _words(prompt), _words(answer)
                if not answer_words:
                    raise ValueError(
                        f"Answer has no word characters: {entry['path']} sample {sample_index}"
                    )
                candidates += 1
                if any(
                    prompt_words[i : i + len(answer_words)] == answer_words
                    for i in range(len(prompt_words) - len(answer_words) + 1)
                ):
                    excluded.append(subject)
                    continue
                rows.append(
                    {
                        "id": f"semantic/{entry['relation']}:{sample_index}:{template_index}",
                        "relation": entry["relation"],
                        "subject": subject,
                        "prompt_prefix": prompt,
                        "answer": answer,
                        "source": {
                            "repository": REPOSITORY,
                            "revision": lock["revision"],
                            "path": entry["path"],
                            "sha256": entry["sha256"],
                            "sample_index": sample_index,
                            "template_index": template_index,
                        },
                    }
                )
    if not rows:
        raise ValueError("No usable semantic geography prompts were found")
    output = root / "datasets/prepared/time-course-mechinterp/prompts.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stable source order and sorted keys make repeated preparation byte-identical.
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    temporary = output.with_suffix(".jsonl.part")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output)
    return {
        "source_files": len(sources),
        "candidates": candidates,
        "excluded": len(excluded),
        "excluded_subjects": excluded,
        "prompts": len(rows),
        "output": str(output),
    }


if __name__ == "__main__":
    try:
        print(json.dumps(prepare_data(), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
