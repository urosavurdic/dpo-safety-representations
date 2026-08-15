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
        mock_base.to.return_value = mock_base
        mock_from_pretrained.return_value = mock_base

        mock_after_m1 = MagicMock()
        mock_after_m2 = MagicMock()

        mock_peft_model.from_pretrained.side_effect = [
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_m1)),
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_m2)),
        ]

        result = load_stage_model("M2")

        calls = mock_peft_model.from_pretrained.call_args_list

        assert calls[0].args == (
            mock_base,
            "urosavurdic/qwen2.5-1.5b-m1-helpful",
        )
        assert calls[1].args == (
            mock_after_m1,
            "urosavurdic/qwen2.5-1.5b-m2-safety",
        )
        assert result == mock_after_m2
        assert result == mock_after_m2


def test_load_stage_model_unknown_stage_raises():
    import pytest
    with pytest.raises(ValueError):
        load_stage_model("M99")


def test_load_stage_model_moves_to_device():
    with patch("src.training.model.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained, \
         patch("src.training.model.PeftModel"), \
         patch("src.training.model.torch.cuda.is_available", return_value=True):
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        load_stage_model("M0")
        mock_model.to.assert_called_once_with("cuda")

def test_load_stage_model_m3_direct_merges_m1_then_direct_dpo_in_order():
    with patch("src.training.model.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained, \
         patch("src.training.model.PeftModel") as mock_peft_model:
        mock_base = MagicMock()
        mock_base.to.return_value = mock_base
        mock_from_pretrained.return_value = mock_base

        mock_after_m1 = MagicMock()
        mock_after_dpo = MagicMock()

        mock_peft_model.from_pretrained.side_effect = [
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_m1)),
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_dpo)),
        ]

        result = load_stage_model("M3_direct")

        calls = mock_peft_model.from_pretrained.call_args_list

        assert calls[0].args == (
            mock_base,
            "urosavurdic/qwen2.5-1.5b-m1-helpful",
        )
        assert calls[1].args == (
            mock_after_m1,
            "urosavurdic/qwen2.5-1.5b-m3-direct-dpo",
        )
        assert result == mock_after_dpo


def test_load_stage_model_m3_direct_alt_merges_m1_alt_then_direct_dpo_in_order():
    """Data-dependence branch: M3_direct_alt must start from M1_alt (Dolly),
    not M1 (Alpaca) - the whole point of the alt branch is a different M1."""
    with patch("src.training.model.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained, \
         patch("src.training.model.PeftModel") as mock_peft_model:
        mock_base = MagicMock()
        mock_base.to.return_value = mock_base
        mock_from_pretrained.return_value = mock_base

        mock_after_m1_alt = MagicMock()
        mock_after_dpo = MagicMock()

        mock_peft_model.from_pretrained.side_effect = [
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_m1_alt)),
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_dpo)),
        ]

        result = load_stage_model("M3_direct_alt")

        calls = mock_peft_model.from_pretrained.call_args_list
        assert calls[0].args == (mock_base, "urosavurdic/qwen2.5-1.5b-m1-alt-helpful")
        assert calls[1].args == (mock_after_m1_alt, "urosavurdic/qwen2.5-1.5b-m3-direct-alt-dpo")
        assert result == mock_after_dpo


def test_load_stage_model_m3_alt_merges_full_three_stage_chain_in_order():
    """M3_alt is the sequential (non-direct) alt-branch endpoint: M1_alt ->
    M2_alt -> M3_alt, mirroring M1->M2->M3 but starting from Dolly."""
    with patch("src.training.model.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained, \
         patch("src.training.model.PeftModel") as mock_peft_model:
        mock_base = MagicMock()
        mock_base.to.return_value = mock_base
        mock_from_pretrained.return_value = mock_base

        mock_after_m1_alt = MagicMock()
        mock_after_m2_alt = MagicMock()
        mock_after_m3_alt = MagicMock()

        mock_peft_model.from_pretrained.side_effect = [
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_m1_alt)),
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_m2_alt)),
            MagicMock(merge_and_unload=MagicMock(return_value=mock_after_m3_alt)),
        ]

        result = load_stage_model("M3_alt")

        calls = mock_peft_model.from_pretrained.call_args_list
        assert calls[0].args == (mock_base, "urosavurdic/qwen2.5-1.5b-m1-alt-helpful")
        assert calls[1].args == (mock_after_m1_alt, "urosavurdic/qwen2.5-1.5b-m2-alt-safety")
        assert calls[2].args == (mock_after_m2_alt, "urosavurdic/qwen2.5-1.5b-m3-alt-dpo")
        assert result == mock_after_m3_alt