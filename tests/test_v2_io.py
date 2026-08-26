import json

from src.v2_io import (
    canonical_json,
    identity_snapshot,
    sha256_bytes,
    write_json_lf,
)


def test_identity_snapshot():
    rows = [
        {
            "record_id": "r1",
            "prompt": "hello",
            "quadrant": "A",
            "source": "HarmBench",
            "split": "direction_estimation",
        }
    ]

    assert identity_snapshot(rows) == [
        {
            "record_id": "r1",
            "prompt": "hello",
            "scored_prompt": None,
            "quadrant": "A",
            "source": "HarmBench",
            "source_dataset": None,
            "c_construction": None,
            "split": "direction_estimation",
        }
    ]


def test_canonical_json_is_key_order_independent():
    first = {"a": 1, "b": 2}
    second = {"b": 2, "a": 1}

    assert canonical_json(first) == canonical_json(second)
    assert sha256_bytes(canonical_json(first)) == (
        sha256_bytes(canonical_json(second))
    )


def test_write_json_is_lf(tmp_path):
    path = tmp_path / "result.json"
    write_json_lf(path, {"text": "ž"})

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert json.loads(raw.decode("utf-8")) == {"text": "ž"}
