"""Did each arm learn what it was trained on? Held-out PKU-SafeRLHF pairs.

For every held-out pair (same safety-disagreement filter as the training set,
no prompt overlap with it) we teacher-force the chosen and the rejected
response and sum log-probabilities over the **response tokens only**, using the
same chat rendering as training (``format_chat_example``): the response tokens
are everything after ``apply_chat_template([user], add_generation_prompt=True)``.

From those sums:

* ``nll_chosen`` / ``nll_rejected``: mean per-token NLL. SFT that learned its
  targets lowers ``nll_chosen`` below the start checkpoint's.
* ``margin``: length-normalised log p(chosen) - log p(rejected), and the share
  of pairs where it is positive.
* ``implicit_reward``: beta * [(log pi(c) - log pi_ref(c)) - (log pi(r) -
  log pi_ref(r))], DPO's own quantity, against the start checkpoint as
  reference, with its accuracy.
"""
from __future__ import annotations

import numpy as np

DPO_BETA = 0.1
MAX_LENGTH = 1024


def _response_spans(tokenizer, prompt: str, response: str):
    from src.training.formatting import format_chat_example

    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    full = format_chat_example({"prompt": prompt, "response": response}, tokenizer)["text"]
    if not full.startswith(prefix):
        raise ValueError("chat template prefix is not a prefix of the training rendering")
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:MAX_LENGTH]
    return full_ids, len(prefix_ids)


def sequence_logprobs(model, tokenizer, prompts, responses, device, batch_size=4):
    """``(sum_logprob, n_tokens)`` of each response given its prompt."""
    import torch

    items = [_response_spans(tokenizer, p, r) for p, r in zip(prompts, responses)]
    sums = np.zeros(len(items))
    counts = np.zeros(len(items), dtype=int)
    pad = tokenizer.pad_token_id
    order = np.argsort([len(ids) for ids, _ in items])
    for start in range(0, len(items), batch_size):
        batch = [int(i) for i in order[start:start + batch_size]]
        width = max(len(items[i][0]) for i in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, i in enumerate(batch):
            seq = items[i][0]
            ids[row, :len(seq)] = torch.tensor(seq)
            mask[row, :len(seq)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        token_logp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu().numpy()
        for row, i in enumerate(batch):
            seq, n_prefix = items[i]
            # position t predicts token t+1; response tokens are seq[n_prefix:]
            span = token_logp[row, n_prefix - 1:len(seq) - 1]
            sums[i], counts[i] = float(span.sum()), len(span)
    return sums, counts


def pair_scores(model, tokenizer, pairs, device, batch_size=4) -> dict:
    prompts = [p["prompt"] for p in pairs]
    c_sum, c_n = sequence_logprobs(model, tokenizer, prompts, [p["chosen"] for p in pairs],
                                   device, batch_size)
    r_sum, r_n = sequence_logprobs(model, tokenizer, prompts, [p["rejected"] for p in pairs],
                                   device, batch_size)
    return {"chosen_sum": c_sum.tolist(), "chosen_n": c_n.tolist(),
            "rejected_sum": r_sum.tolist(), "rejected_n": r_n.tolist()}


def summarise(scores: dict, ref: dict | None = None, beta: float = DPO_BETA) -> dict:
    c_sum, c_n = np.asarray(scores["chosen_sum"]), np.asarray(scores["chosen_n"])
    r_sum, r_n = np.asarray(scores["rejected_sum"]), np.asarray(scores["rejected_n"])
    margin = c_sum / c_n - r_sum / r_n
    out = {
        "n_pairs": int(len(c_sum)),
        "nll_chosen": float(-(c_sum.sum() / c_n.sum())),
        "nll_rejected": float(-(r_sum.sum() / r_n.sum())),
        "margin_mean": float(margin.mean()),
        "margin_positive_share": float((margin > 0).mean()),
    }
    if ref is not None:
        rc, rr = np.asarray(ref["chosen_sum"]), np.asarray(ref["rejected_sum"])
        reward = beta * ((c_sum - rc) - (r_sum - rr))
        out.update({
            "implicit_reward_mean": float(reward.mean()),
            "implicit_reward_accuracy": float((reward > 0).mean()),
            "delta_logp_chosen_mean": float((c_sum - rc).mean()),
            "delta_logp_rejected_mean": float((r_sum - rr).mean()),
        })
    return out
