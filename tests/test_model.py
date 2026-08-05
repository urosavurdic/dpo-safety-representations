from src.training.model import create_lora_config


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