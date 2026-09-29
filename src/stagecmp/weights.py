"""Weight-space comparison between checkpoints, from the LoRA rank-r factors.

``merge_and_unload()`` adds ``(alpha/r) B A`` into the frozen base weight, so a
chain composes by addition and the difference between any two of the nine
checkpoints is exact::

    W_stage       = W_0 + SUM over chain(stage) of s * B_i A_i,   s = alpha/r
    dW(X -> Y)    = SUM over chain(Y) - SUM over chain(X)

which is a short list of SIGNED low-rank terms. Nothing here ever materialises a
dense ``dW``: every quantity reduces to r-by-r Gram matrices.

    ||B A||_F^2            = tr( (B^T B) (A A^T) )
    <B1 A1, B2 A2>_F       = tr( (B1^T B2) (A2 A1^T) )

For a sum of terms, expand bilinearly over those pairwise inner products.
Column spaces come from a factored thin SVD: with ``B_cat = Q_B R_B`` and
``A_cat^T = Q_A R_A``, ``dW = Q_B (R_B R_A^T) Q_A^T``, so the left singular
vectors are ``Q_B U`` where ``U`` comes from the small k-by-k SVD.

**The control that must accompany any subspace claim.** Each term has rank at
most r=64 out of 1536, because that is what was imposed at training time. Two
INDEPENDENT random rank-64 subspaces in 1536 dimensions are already nearly
orthogonal, so "the two updates occupy different subspaces" is the null, not a
finding. ``random_subspace_angles`` supplies that baseline and
``compare_updates`` reports it alongside every principal-angle number.

This module is written but deliberately NOT executed against the real adapters:
that needs ~1.1 GB of downloads. ``load_adapter`` reads the local HuggingFace
cache and refuses to fetch unless explicitly told to.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src.training.model import STAGE_ADAPTER_CHAINS

#: LoRA config used for every stage in this project (configs/*.yaml)
LORA_R = 64
LORA_ALPHA = 128
SCALING = LORA_ALPHA / LORA_R          # = 2.0

RESIDUAL_WRITERS = ("o_proj", "down_proj")
ALL_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")

_KEY = re.compile(r"layers\.(\d+)\.[^.]*\.([a-z_]+_proj)\.lora_([AB])\.weight")


@dataclass
class LowRankTerm:
    """One signed ``sign * scaling * B @ A``."""
    sign: float
    b: np.ndarray            # (out, r)
    a: np.ndarray            # (r, in)
    scaling: float = SCALING
    source: str = ""

    @property
    def coef(self) -> float:
        return self.sign * self.scaling


@dataclass
class Update:
    """``dW`` for one (pair, layer, module) as a list of signed terms."""
    terms: list[LowRankTerm] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.terms)

    def dense(self) -> np.ndarray:
        """Only for tests and tiny shapes. Never call this on a real adapter."""
        return sum(t.coef * (t.b @ t.a) for t in self.terms)


# ------------------------------------------------------------- adapter I/O -- #

def load_adapter(repo_id: str, allow_download: bool = False) -> dict:
    """LoRA tensors for one adapter, from the local HF cache by default.

    ``allow_download=False`` keeps this module runnable offline and makes the
    ~1.1 GB fetch an explicit choice rather than a side effect of importing.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    path = hf_hub_download(
        repo_id=repo_id,
        filename="adapter_model.safetensors",
        local_files_only=not allow_download,
    )
    return load_file(path)


def parse_adapter(state: dict) -> dict:
    """``{(layer, module): {'A': ndarray, 'B': ndarray}}``."""
    out: dict = {}
    for key, tensor in state.items():
        m = _KEY.search(key)
        if not m:
            continue
        layer, module, which = int(m.group(1)), m.group(2), m.group(3)
        out.setdefault((layer, module), {})[which] = np.asarray(tensor, dtype=np.float64)
    incomplete = [k for k, v in out.items() if set(v) != {"A", "B"}]
    if incomplete:
        raise KeyError(
            f"{len(incomplete)} module(s) missing an A or B factor, e.g. {incomplete[:3]}. "
            "The adapter's key naming does not match what this module assumes."
        )
    return out


def chain_updates(stage: str, adapters: dict) -> dict:
    """``{(layer, module): Update}`` for a stage's whole chain from the base."""
    if stage not in STAGE_ADAPTER_CHAINS:
        raise ValueError(f"unknown stage {stage!r}")
    acc: dict = {}
    for repo in STAGE_ADAPTER_CHAINS[stage]:
        for slot, factors in adapters[repo].items():
            acc.setdefault(slot, Update()).terms.append(
                LowRankTerm(sign=+1.0, b=factors["B"], a=factors["A"], source=repo)
            )
    return acc


def difference(stage_x: str, stage_y: str, adapters: dict) -> dict:
    """``dW(X -> Y)`` per slot, as signed low-rank terms. Exact."""
    ux, uy = chain_updates(stage_x, adapters), chain_updates(stage_y, adapters)
    out: dict = {}
    for slot in set(ux) | set(uy):
        terms = list(uy.get(slot, Update()).terms)
        for t in ux.get(slot, Update()).terms:
            terms.append(LowRankTerm(sign=-t.sign, b=t.b, a=t.a, source=t.source))
        out[slot] = Update(terms)
    return out


# ------------------------------------------------------- low-rank algebra -- #

def inner(u: Update, v: Update) -> float:
    """Frobenius inner product, via r-by-r Grams only."""
    total = 0.0
    for tu in u.terms:
        for tv in v.terms:
            total += tu.coef * tv.coef * float(
                np.trace((tu.b.T @ tv.b) @ (tv.a @ tu.a.T))
            )
    return total


def frobenius_norm(u: Update) -> float:
    return float(np.sqrt(max(inner(u, u), 0.0)))


def cosine(u: Update, v: Update) -> float:
    nu, nv = frobenius_norm(u), frobenius_norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return inner(u, v) / (nu * nv)


def _factored(u: Update) -> tuple[np.ndarray, np.ndarray]:
    b_cat = np.concatenate([t.coef * t.b for t in u.terms], axis=1)
    a_cat = np.concatenate([t.a for t in u.terms], axis=0)
    return b_cat, a_cat


def column_basis(u: Update, rank: int | None = None, tol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis for ``col(dW)``, without forming ``dW``."""
    b_cat, a_cat = _factored(u)
    q_b, r_b = np.linalg.qr(b_cat)
    q_a, r_a = np.linalg.qr(a_cat.T)
    small = r_b @ r_a.T
    left, sv, _ = np.linalg.svd(small, full_matrices=False)
    keep = sv > (tol * (sv[0] if sv.size else 1.0))
    if rank is not None:
        keep &= np.arange(sv.size) < rank
    return q_b @ left[:, keep]


def singular_values(u: Update) -> np.ndarray:
    b_cat, a_cat = _factored(u)
    _q_b, r_b = np.linalg.qr(b_cat)
    _q_a, r_a = np.linalg.qr(a_cat.T)
    return np.linalg.svd(r_b @ r_a.T, compute_uv=False)


def principal_angles_deg(basis_a: np.ndarray, basis_b: np.ndarray) -> dict:
    sv = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))
    return {"mean_deg": float(angles.mean()), "min_deg": float(angles.min()),
            "max_deg": float(angles.max()),
            "per_component_deg": [float(a) for a in angles]}


def effective_rank(sv: np.ndarray) -> float:
    s2 = np.asarray(sv, float) ** 2
    total = s2.sum()
    if total <= 0:
        return float("nan")
    p = s2[s2 > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def random_subspace_angles(
    hidden: int = 1536, rank: int = LORA_R, n_samples: int = 50, seed: int = 20260904
) -> dict:
    """Principal angles between two INDEPENDENT random rank-``rank`` subspaces.

    The null any "different subspaces" reading must be compared against: at
    rank 64 in 1536 dimensions, unrelated subspaces are already nearly
    orthogonal, so a large angle on its own says nothing.
    """
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_samples):
        qa, _ = np.linalg.qr(rng.normal(size=(hidden, rank)))
        qb, _ = np.linalg.qr(rng.normal(size=(hidden, rank)))
        means.append(principal_angles_deg(qa, qb)["mean_deg"])
    means = np.asarray(means)
    return {"rank": rank, "hidden": hidden, "n_samples": n_samples, "seed": seed,
            "mean_deg": float(means.mean()),
            "p2.5": float(np.percentile(means, 2.5)),
            "p97.5": float(np.percentile(means, 97.5))}


def compare_updates(u: Update, v: Update, base_norm: float | None = None) -> dict:
    """Every weight-space quantity for one slot, with the random control."""
    sv_u, sv_v = singular_values(u), singular_values(v)
    basis_u, basis_v = column_basis(u), column_basis(v)
    out = {
        "frobenius": {"u": frobenius_norm(u), "v": frobenius_norm(v)},
        "cosine": cosine(u, v),
        "principal_angles": principal_angles_deg(basis_u, basis_v),
        "effective_rank": {"u": effective_rank(sv_u), "v": effective_rank(sv_v)},
        "rank": {"u": int(basis_u.shape[1]), "v": int(basis_v.shape[1])},
        "random_subspace_control": random_subspace_angles(
            hidden=basis_u.shape[0], rank=min(basis_u.shape[1], basis_v.shape[1])
        ),
        "reading": (
            "Rank is bounded by the LoRA r=64 imposed at training time, so a "
            "large principal angle is only informative against the random "
            "control reported here. Comparisons between updates taken at "
            "DIFFERENT base points (e.g. M2->M3 vs M1->M3_direct) are not "
            "like-for-like; only arms sharing a start checkpoint are."
        ),
    }
    if base_norm:
        out["frobenius_relative"] = {k: v / base_norm for k, v in out["frobenius"].items()}
    return out
