# Stochastic Rounding and Update Sparsity in BF16 Reinforcement Learning with SGD

Official PyTorch implementation of **Maximally Coupled Stochastic Rounding (`MCSR-L`)** and **Coupled Master-SR** for BF16 Reinforcement Learning with Verifiable Rewards (RLVR).

## Overview

In BF16 RLVR post-training, standard round-to-nearest-even (`Master-RTNE`) leaves ~99% of per-step BF16 serving weights unchanged—enabling sparse delta weight synchronization—but suffers weight sticking and policy lag. Replacing RTNE with independent per-step stochastic rounding (`Master-SR`) restores unbiased updates, yet incurs an exact stasis flip probability of `2 q_i (1 - q_i)` on off-grid master weights (and deterministic error feedback incurs an even higher limit-cycle flip rate, `50.21%` vs. `33.35%` in static CPU replay), inflating per-step bitwise weight changes by 6–10×.

This repository provides:
- **`bf16_sr_core/mcsr_step.py`**: Maximally coupled stochastic rounding across non-uniform BF16 binades (`Coupled Master-SR` at `6p` bytes and bit-packed 4-bit sub-ULP `MCSR-4` at `2.5p` bytes with `L = 8` and `8`-bit `MCSR-8` at `3p` bytes with `L = 128`), achieving minimal bitwise change probability ($\mathrm{TV}$) and minimal expected absolute displacement ($\mathcal{W}_1$) with zero reverse-direction moves.
- **`bf16_sr_core/bf16_sr.py` & `sr_step.py`**: Bitwise BF16 stochastic rounding and in-place masterless `2p` BF16-SR updates.
- **`bf16_sr_core/delta_wire.py`**: Bit-exact sparse delta encoding and SHA-256 verification utilities.
- **`experiments/verify_static_ef_cpu.py`**: 10-second CPU verification of the static error-feedback vs. independent SR vs. coupled SR flip rates (20,000,000 comparisons) and BF16 binade-crossing counterexamples.
- **`experiments/run_matched_ood_v22.py` & `run_countdown_mcsr_v22.py`**: Closed-loop GRPO RLVR training and evaluation runners for warm-start math reasoning (MATH-500 / GSM8K) and cold-start Countdown-3.

## Quick Start (CPU Verification, < 10 Seconds)

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/experiments"

# 1. Exact rational verification (1,681 grid pairs) + BF16 coupling & MCSR-4 unit tests
python bf16_sr_core/test_mcsr.py

# 2. Closed-loop runner & E2/E3/E4a unit tests
python -m unittest discover -s experiments -p "test_*.py" -v

# 3. 20M-comparison static Error Feedback vs. SR vs. Coupled SR CPU verification
python experiments/verify_static_ef_cpu.py
```

## Minimal Usage (`MCSR-4` & `Coupled Master-SR`)

```python
import torch
from bf16_sr_core.mcsr_step import MCSR, MASTER

params = [p for p in model.parameters() if p.requires_grad]

# 2.5p Masterless MCSR-4 (4-bit packed signed sub-ULP residual, L=8)
state = MCSR(params, L=8)
# Or 6p Coupled Master-SR (L=inf FP32 master with maximal coupling):
# state = MCSR(params, L=MASTER)

# After backward() and gradient clipping:
grads = [p.grad for p in params]
stats = state.update_(grads, seed=42, step=step, alpha=-0.1)
# stats reports exact bit_changed_count, reverse_flip_count (0), saturated_count, and packed state_bytes
```

## Repository Layout

```text
bf16-sr-rlvr/
├── bf16_sr_core/
│   ├── bf16_sr.py                  # Bitwise BF16 stochastic rounding primitives
│   ├── delta_wire.py               # Sparse COO/delta serialization & SHA-256 checks
│   ├── mcsr_step.py                # MCSR-L (L=8, 128) & Coupled Master-SR (L=inf)
│   ├── sampling_utils.py           # Exact uniform/integer sampling helpers
│   ├── sr_step.py                  # Masterless 2p BF16-SR step
│   └── test_mcsr.py                # Exact rational & grid coupling tests
├── experiments/
│   ├── run_matched_ood_v22.py      # Warm-start MATH-500 / GSM8K GRPO runner (SGD & AdamW)
│   ├── run_countdown_mcsr_v22.py   # Cold-start Countdown-3 GRPO runner
│   ├── bench_system_accounting_v1.py # Isolated GPU memory & COO sync benchmark
│   ├── verify_static_ef_cpu.py     # Static EF / SR / Coupled SR CPU verifier
│   └── test_*.py                   # Regression & Phase 0 unit tests
├── requirements.txt
└── LICENSE
```

## License

MIT License. See `LICENSE` for details.
