#!/usr/bin/env python3
"""CPU verification of static BF16 error feedback and two rounding controls."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from run_matched_ood_v22 import atomic_write_json, create_mcsr_state, sr_into


def verify_two_point_grid(numerators, denominator):
    # Integer units of h / (4 * denominator) make the simulation exact.
    scale = 4 * denominator
    q = 4 * np.array(numerators, dtype=np.int64)
    starts = np.arange(-2, 3)[None, :, None] * denominator
    residual = np.broadcast_to(starts, (2, 5, q.size)).copy()
    upper_is_even = np.array([False, True])[:, None, None]

    def advance(r):
        s = r + q
        upper = (2 * s > scale) | ((2 * s == scale) & upper_is_even)
        return s - scale * upper, upper

    # The first two transitions also handle an initial tie at either endpoint.
    residual, _ = advance(residual)
    residual, prev = advance(residual)
    start_r, start_word = residual, prev
    changes = np.zeros_like(residual)
    for _ in range(denominator):
        residual, upper = advance(residual)
        changes += upper != prev
        prev = upper
    assert np.array_equal(residual, start_r) and np.array_equal(prev, start_word)
    expected = 2 * np.minimum(numerators, denominator - np.array(numerators))
    assert np.all(changes == expected)
    assert np.all(expected * denominator > 2 * np.array(numerators) * (denominator - np.array(numerators)))
    return changes[0, 0]


def find_cycle(transition, init, max_steps):
    seen, outputs = {}, []
    state = init
    for step in range(max_steps):
        if state in seen:
            start = seen[state]
            cycle = outputs[start:]
            flips = sum(a != b for a, b in zip(cycle, cycle[1:] + cycle[:1]))
            return start, cycle, flips
        seen[state] = step
        state, out = transition(state)
        outputs.append(out)
    raise AssertionError("residual did not cycle within the planned replay")


def fp32_cycle(a, h, q):
    master = torch.tensor(float(a + q * h), dtype=torch.float32, device="cpu")
    assert Fraction(master.item()) == a + q * h

    def advance(residual_bits):
        residual = torch.tensor(residual_bits, dtype=torch.int32).view(torch.float32)
        temp = master + residual
        word = temp.to(torch.bfloat16)
        residual = temp - word.float()
        return residual.view(torch.int32).item(), word.view(torch.int16).item()

    return find_cycle(advance, 0, max_steps=100_000)


def verify_counterexamples():
    h = Fraction(1, 128)
    a, q = Fraction(1), Fraction(1, 16)
    # These are consecutive real BF16 grid points, with the corresponding raw words.
    grid = {0x3F7E: a - h, 0x3F7F: a - h / 2, 0x3F80: a,
            0x3F81: a + h, 0x3F82: a + 2 * h}

    def exact_binade_step(residual):
        temp = a + q * h + residual
        assert min(grid.values()) <= temp <= max(grid.values())
        word = min(grid, key=lambda k: (abs(grid[k] - temp), k & 1))
        return temp - grid[word], word

    start, cycle, flips = find_cycle(exact_binade_step, Fraction(0), max_steps=32)
    assert (len(cycle), flips) == (8, 3)
    assert (cycle.count(0x3F80), cycle.count(0x3F81), cycle.count(0x3F7F)) == (6, 1, 1)
    assert fp32_cycle(a, h, q) == (start, cycle, flips)
    binade = {"a": str(a), "h": str(h), "q": str(q), "transient_steps": start,
              "cycle": [str(grid[word]) for word in cycle], "cycle_steps": len(cycle),
              "flips": flips, "flip_rate": str(Fraction(flips, len(cycle))),
              "two_point_flip_rate": str(2 * q)}

    a, q = Fraction(2) - h, Fraction(60001, 65536)
    start, cycle, flips = fp32_cycle(a, h, q)
    assert set(cycle) == {0x3FFF, 0x4000}
    assert (len(cycle), flips) == (8193, 1384)
    assert 2 * (1 - q) == Fraction(5535, 32768)
    fp32 = {"a": str(a), "h": str(h), "q": str(q), "transient_steps": start,
            "cycle_steps": len(cycle), "flips": flips,
            "flip_rate": str(Fraction(flips, len(cycle))),
            "exact_two_point_flip_rate": str(2 * (1 - q))}
    return binade, fp32


def simulate_static_sync(seed=20271123, n_per_distribution=50_000, steps=200):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Reject exactly on-grid FP32 values rather than perturbing the distribution.
    n = n_per_distribution + 1024
    normal = torch.randn(n, dtype=torch.float32, generator=gen)
    exponent = torch.rand(n, generator=gen) * 120 - 60
    signs = 2 * torch.randint(2, (n,), generator=gen) - 1
    log_uniform = torch.exp2(exponent) * signs
    off_grid = [x[(x.view(torch.int32) & 65535) != 0][:n_per_distribution]
                for x in (normal, log_uniform)]
    assert all(x.numel() == n_per_distribution for x in off_grid)
    master = torch.cat(off_grid)

    def ef_cast(residual):
        # Fair Cast, §2, local TeX lines 304–315: FP32 addition, BF16 RTNE,
        # then subtraction in FP32 to update the error-feedback residual.
        temp = master + residual
        serving = temp.to(torch.bfloat16)
        return serving, temp - serving.float()

    ef, residual = ef_cast(torch.zeros_like(master))
    sr = torch.empty_like(master, dtype=torch.bfloat16)
    sr_into(master, sr, gen)
    coupled = sr.clone()
    coupled_initial = coupled.clone()
    state = create_mcsr_state("M_coupled", [coupled])
    state.restore_([{"weight": coupled_initial, "master": master, "residual": None}])
    zero = torch.zeros_like(master)
    counts = torch.zeros((3, 2), dtype=torch.int64)
    for step in range(1, steps + 1):
        next_ef, residual = ef_cast(residual)
        next_sr = torch.empty_like(sr)
        sr_into(master, next_sr, gen)
        for i, (old, new) in enumerate(((ef, next_ef), (sr, next_sr))):
            changed = old.view(torch.int16) != new.view(torch.int16)
            counts[i] += changed.view(2, n_per_distribution).sum(1)
        stats = state.update_([zero], seed=seed, step=step)
        assert stats["bit_changed_count"] == stats["reverse_flip_count"] == 0
        ef, sr = next_ef, next_sr

    final = state.checkpoint()[0]
    assert torch.equal(coupled_initial.view(torch.int16), final["weight"].view(torch.int16))
    assert torch.equal(master.view(torch.int32), final["master"].view(torch.int32))
    q = (master.view(torch.int32) & 65535).double() / 65536
    rates = counts.double() / (n_per_distribution * steps)
    assert (rates[0] > rates[1]).all() and (rates[2] == 0).all()

    def summary(indices):
        totals = counts[:, indices].sum(1)
        denom = n_per_distribution * len(indices) * steps
        return {"comparisons": denom, "flips": totals.tolist(),
                "flip_rate": (totals.double() / denom).tolist(),
                "independent_theory": (2 * q * (1 - q)).view(2, -1)[indices].mean().item()}

    return {"seed": seed, "device": "cpu", "weights": master.numel(), "sync_steps_after_initial_cast": steps,
            "distributions": {"normal": "N(0,1)", "signed_log_uniform": "sign * 2**Uniform(-60,60)"},
            "method_order": ["ef_cast_bf16", "sr_cast_bf16", "coupled_master_sr"],
            "normal": summary([0]), "signed_log_uniform": summary([1]), "pooled": summary([0, 1])}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    q = np.arange(1, 2048, 2, dtype=np.int64)
    flips = verify_two_point_grid(q, 2048)
    for denominator in (31, 63, 127, 255):
        verify_two_point_grid(np.arange(1, denominator, dtype=np.int64), denominator)
    binade, fp32 = verify_counterexamples()
    result = {
        "exact_two_point": {
            "q_values_checked": q.size + sum(d - 1 for d in (31, 63, 127, 255)),
            "dyadic_midpoint_grid_size": q.size,
            "initial_residuals_over_h": ["-1/2", "-1/4", "0", "1/4", "1/2"],
            "ties_to_even_endpoints_checked": ["lower", "upper"],
            "ef_flip_rate_formula": "min(2*q, 2*(1-q))",
            "sr_flip_rate_formula": "2*q*(1-q)",
            "midpoint_grid_average": {"ef": float(flips.mean() / 2048),
                                      "independent_sr": float(np.mean(2 * (q / 2048) * (1 - q / 2048)))},
            "continuous_uniform_q_average": {"ef": "1/2", "independent_sr": "1/3", "coupled": "0"},
        },
        "bf16_binade_crossing": binade,
        "fp32_addition_shift": fp32,
        "static_cpu_simulation": simulate_static_sync(),
    }
    path = Path(__file__).with_name("res_static_ef_cpu_verify.json")
    atomic_write_json(str(path), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
