from src.data_pipeline.build_eval_set import build_quadrant_c, QUADRANT_C_PROMPTS


def test_quadrant_c_shape():
    records = build_quadrant_c()
    assert len(records) == len(QUADRANT_C_PROMPTS)
    for r in records:
        assert set(r.keys()) == {"prompt", "quadrant", "source", "category"}
        assert r["quadrant"] == "C"
        assert isinstance(r["prompt"], str) and len(r["prompt"]) > 0


def test_quadrant_c_prompts_are_unique():
    assert len(set(QUADRANT_C_PROMPTS)) == len(QUADRANT_C_PROMPTS)