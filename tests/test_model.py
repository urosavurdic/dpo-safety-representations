from unittest.mock import patch, MagicMock

from src.training.model import create_lora_config, load_model, load_stage_model


def test_lora_config():
    cfg = {
        "lora": {
            "r": 64,
            "alpha": 128,
            "dropout": 0.05,
        }
    }
    lora = create_lora_config(cfg)
    assert lora.r == 64
    assert lora.lora_alpha == 128
    assert lora.lora_dropout == 0.05


def test_load_model_no_adapter_skips_merge():
    cfg = {"model": {"name": "fake-model", "trust_remote_code": True}}
    with patch("src.training.model.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained, \
         patch("src.training.model.PeftModel") as mock_peft_model:
        mock_from_pretrained.return_value = MagicMock()
        load_model(cfg)
        mock_from_pretrained.assert_called_once()
        mock_peft_model.from_pretrained.assert_not_called()


def test_load_model_with_adapter_merges():
    cfg = {
        "model": {
            "name": "fake-model",
            "trust_remote_code": True,
            "init_from_adapter": "fake-org/fake-adapter",
        }
    }
    with patch("src.training.model.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained, \
         patch("src.training.model.PeftModel") as mock_peft_model:
        mock_base_model = MagicMock()
        mock_from_pretrained.return_value = mock_base_model
        mock_merged = MagicMock()
        mock_peft_model.from_pretrained.return_value.merge_and_unload.return_value = mock_merged

        result = load_model(cfg)

        mock_peft_model.from_pretrained.assert_called_once_with(mock_base_model, "fake-org/fake-adapter")
        assert result == mock_merged

def test_load_stage_model_m0_no_adapters():
    with patch("src.training.model.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained, \
         patch("src.training.model.PeftModel") as mock_peft_model:
        mock_from_pretrained.return_value = MagicMock()
        load_stage_model("M0")
        mock_peft_model.from_pretrained.assert_not_called()


def test_load_stage_model_m2_merges_m1_then_m2_in_order():
    with patch("src.training.model.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained, \
         patch("src.training.model.PeftModel") as mock_peft_model:
        mock_base = MagicMock()
        mock_from_pretrained.return_value = mock_base
        mock_after_m1 = MagicMock()
        mock_after_m2 = MagicMock()
        mock_peft_model.from_pretrained.side_effect = [
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_m1)),
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_m2)),
        ]
        result = load_stage_model("M2")
        calls = mock_peft_model.from_pretrained.call_args_list
        assert calls[0].args == (mock_base, "urosavurdic/qwen2.5-1.5b-m1-helpful")
        assert calls[1].args == (mock_after_m1, "urosavurdic/qwen2.5-1.5b-m2-safety")
        assert result == mock_after_m2


def test_load_stage_model_unknown_stage_raises():
    import pytest
    with pytest.raises(ValueError):
        load_stage_model("M99")

