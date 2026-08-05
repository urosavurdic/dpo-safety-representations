"""
Shared training utilities for SFT and DPO.

Everything in this package is intentionally model-agnostic so that
M1 (Helpful SFT), M2 (Safety SFT), and later M3 (DPO) share as much
infrastructure as possible.
"""