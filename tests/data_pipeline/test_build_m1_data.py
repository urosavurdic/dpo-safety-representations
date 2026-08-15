from src.data_pipeline.build_m1_data import (
    build_m1_dataset,
    remove_flagged_prompts,
    update_exclusion_list,
    load_exclusion_list,
    SOURCES,
)

def _fake_row(instruction, output, input_=""):
    return {"instruction": instruction, "input": input_, "output": output}


def test_excludes_reserved_prompts():
    rows = [
        _fake_row("Reserved question", "some answer"),
        _fake_row("Different question", "another answer"),
    ]
    reserved = ["Reserved question"]
    result = build_m1_dataset(rows, reserved, n_target=10)
    prompts = [r["prompt"] for r in result]
    assert "Reserved question" not in prompts
    assert "Different question" in prompts


def test_exclusion_is_case_and_whitespace_insensitive():
    rows = [_fake_row("  Reserved Question  ", "answer")]
    reserved = ["reserved question"]
    result = build_m1_dataset(rows, reserved, n_target=10)
    assert result == []


def test_filters_multiturn_and_empty_rows():
    rows = [
        _fake_row("has input", "answer", input_="some context"),
        _fake_row("", "answer"),
        _fake_row("valid question", ""),
        _fake_row("valid question 2", "valid answer"),
    ]
    result = build_m1_dataset(rows, reserved_prompts=[], n_target=10)
    assert len(result) == 1
    assert result[0]["prompt"] == "valid question 2"


def test_respects_n_target():
    rows = [_fake_row(f"q{i}", f"a{i}") for i in range(10)]
    result = build_m1_dataset(rows, reserved_prompts=[], n_target=3)
    assert len(result) == 3

def test_remove_flagged_prompts():
    data = [
        {"prompt": "Write a sentence to explain the process of photosynthesis.", "response": "x"},
        {"prompt": "Something unrelated", "response": "y"},
    ]
    flagged = ["Write a sentence to explain the process of photosynthesis."]
    result = remove_flagged_prompts(data, flagged)
    assert len(result) == 1
    assert result[0]["prompt"] == "Something unrelated"


def test_update_exclusion_list_merges_and_dedupes(tmp_path):
    path = tmp_path / "exclusions.json"
    first = update_exclusion_list(str(path), ["Prompt A", "Prompt B"])
    assert set(first) == {"Prompt A", "Prompt B"}

    second = update_exclusion_list(str(path), ["prompt a", "Prompt C"])  # case-insensitive dup of A
    assert len(second) == 3
    assert "Prompt C" in second

    reloaded = load_exclusion_list(str(path))
    assert len(reloaded) == 3


def test_dolly_normalizer_maps_to_common_shape_reused_by_build_m1_dataset():
    """M1_alt's source (Dolly: instruction/context/response/category) must
    normalize to the exact same {instruction, input, output} shape Alpaca
    rows already have, so build_m1_dataset (selection/exclusion/dedup logic)
    is reused completely unchanged - not reimplemented per source."""
    dolly_row = {
        "instruction": "What is the capital of France?",
        "context": "",
        "response": "Paris.",
        "category": "closed_qa",
    }
    normalized = SOURCES["dolly"]["normalize_row"](dolly_row)
    assert normalized == {"instruction": "What is the capital of France?", "input": "", "output": "Paris."}

    # Multi-turn-equivalent (non-empty context) rows must still be filterable
    # by build_m1_dataset's existing `row["input"] != ""` single-turn check.
    dolly_multiturn_row = {
        "instruction": "Summarize this.",
        "context": "Some long passage...",
        "response": "A summary.",
        "category": "summarization",
    }
    result = build_m1_dataset(
        [SOURCES["dolly"]["normalize_row"](dolly_row), SOURCES["dolly"]["normalize_row"](dolly_multiturn_row)],
        reserved_prompts=[],
        n_target=10,
    )
    prompts = [r["prompt"] for r in result]
    assert "What is the capital of France?" in prompts
    assert "Summarize this." not in prompts  # excluded: non-empty context/input


def test_alpaca_normalizer_is_identity_preserving_backward_compatibility():
    row = {"instruction": "q", "input": "", "output": "a"}
    assert SOURCES["alpaca"]["normalize_row"](row) == row