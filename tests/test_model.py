from unittest.mock import patch, MagicMock

from src.training.model import create_lora_config, load_model


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