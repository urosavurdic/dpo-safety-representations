"""
Full-fine-tuning DPO smoke test: does a full-parameter DPO step fit on a
T4 (16GB), and how fast is it?

NOT a real training run, NOT wired through TRL's DPOTrainer. This mimics
DPO's computational SHAPE (two forward passes through a trainable
"policy", two through a frozen "reference", backward on policy only, one
optimizer step) to measure real memory/speed -- it does not compute a
correct DPO loss and isn't meant to train anything usable.

Why a separate reference model is needed here, unlike M3's actual LoRA
training: TRL's DPOTrainer only gets the free "disable adapter" trick for
PEFT models. Per TRL's own source, full fine-tuning needs an explicit
ref_model -- a second full frozen copy resident in memory alongside the
trainable policy. That's the real bottleneck this measures.

If this fits: the real run needs train_dpo.py modified to pass an
explicit ref_model to DPOTrainer instead of ref_model=None -- not written
here, since that requires seeing train_dpo.py's current structure first.
"""
import time

import torch
from torch.optim import AdamW

from src.training.model import load_stage_model

SEQ_LEN = 256  # representative of a real prompt+response pair length
BATCH_SIZE = 1
N_STEPS = 3
USE_8BIT_ADAM = True


def make_fake_batch(vocab_size, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, device="cuda"):
    # Random token ids -- fine for a memory/speed probe, meaningless for correctness.
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def main():
    assert torch.cuda.is_available(), "This smoke test needs a GPU to be meaningful."
    device = "cuda"

    print("Loading policy (trainable) M2...")
    policy = load_stage_model("M2").to(device)
    policy.gradient_checkpointing_enable()
    policy.train()

    print("Loading reference (frozen) M2 -- a second full copy...")
    reference = load_stage_model("M2").to(device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Trainable params (full FT, no LoRA): {n_params:,}")

    if USE_8BIT_ADAM:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=1e-6)
        print("Using 8-bit AdamW (bitsandbytes).")
    else:
        optimizer = AdamW(policy.parameters(), lr=1e-6)
        print("Using standard fp32 AdamW.")

    vocab_size = policy.config.vocab_size
    torch.cuda.reset_peak_memory_stats()
    step_times = []

    for step in range(N_STEPS):
        t0 = time.time()
        chosen_ids = make_fake_batch(vocab_size, device=device)
        rejected_ids = make_fake_batch(vocab_size, device=device)

        with torch.no_grad():
            ref_chosen = reference(chosen_ids).logits
            ref_rejected = reference(rejected_ids).logits

        policy_chosen = policy(chosen_ids).logits
        policy_rejected = policy(rejected_ids).logits

        # Placeholder loss -- right gradient-flow shape (policy vs. detached
        # reference), NOT a real DPO loss. Only measures memory/compute.
        loss = (
            (policy_chosen - ref_chosen.detach()).pow(2).mean()
            - (policy_rejected - ref_rejected.detach()).pow(2).mean()
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        step_times.append(time.time() - t0)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Step {step}: loss={loss.item():.4f}, time={step_times[-1]:.2f}s, peak_mem={peak_gb:.2f}GB")

    print(f"\nMean step time: {sum(step_times)/len(step_times):.2f}s")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak GPU memory: {peak:.2f}GB (T4 budget: 16GB)")
    if peak > 15:
        print("WARNING: within 1GB of the T4 budget -- a real run (longer "
              "sequences, eval batches, larger batch size) will very likely OOM.")


if __name__ == "__main__":
    main()