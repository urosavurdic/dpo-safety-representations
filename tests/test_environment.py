import sys

def test_python_version():
    assert sys.version_info >= (3, 10), "Need Python 3.10+"

def test_core_imports():
    import torch
    import transformers
    import trl
    import peft
    import sklearn
    import datasets
    # If any of these fail to import, this test fails with a clear traceback
    # pointing at exactly which package is missing/broken.

def test_torch_basic_op():
    import torch
    x = torch.randn(4, 4)
    y = x @ x.T
    assert y.shape == (4, 4)