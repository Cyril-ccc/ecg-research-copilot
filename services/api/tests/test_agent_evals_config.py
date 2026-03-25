from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _select_smoke_tests(items: list[dict], max_tests: int) -> list[dict]:
    selected: list[dict] = []
    seen_ids: set[str] = set()

    def _add(item: dict) -> bool:
        test_id = str(item.get("id", "")).strip() or f"row-{len(seen_ids)}"
        if test_id in seen_ids:
            return False
        selected.append(item)
        seen_ids.add(test_id)
        return len(selected) >= max_tests

    for category in ("normal", "malicious", "oversize", "ambiguous"):
        for item in items:
            if str(item.get("category", "")).strip().lower() != category:
                continue
            if _add(item):
                return selected
            break

    for item in items:
        if _add(item):
            break

    return selected


def test_agent_tests_yaml_has_required_mix_and_unique_ids():
    tests_path = _repo_root() / "evals" / "agent_tests.yaml"
    payload = yaml.safe_load(tests_path.read_text(encoding="utf-8"))

    assert isinstance(payload, list)
    assert len(payload) >= 20

    categories = Counter(str(item.get("category", "")).strip().lower() for item in payload)
    assert categories["normal"] >= 10
    assert categories["ambiguous"] >= 3
    assert categories["malicious"] >= 5
    assert categories["oversize"] >= 2

    ids = [str(item.get("id", "")).strip() for item in payload]
    assert all(ids)
    assert len(ids) == len(set(ids))

    for item in payload:
        expected = item.get("expected")
        assert isinstance(expected, dict)
        status = str(expected.get("status", "")).strip().upper()
        assert status in {"SUCCEEDED", "REJECTED"}


def test_agent_tests_smoke_slice_covers_multiple_categories():
    tests_path = _repo_root() / "evals" / "agent_tests.yaml"
    payload = yaml.safe_load(tests_path.read_text(encoding="utf-8"))

    smoke = _select_smoke_tests(payload, 5)
    categories = {str(item.get("category", "")).strip().lower() for item in smoke}
    assert "normal" in categories
    assert "malicious" in categories
    assert "oversize" in categories
