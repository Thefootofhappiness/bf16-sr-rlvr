"""CPU regression, finite-grid oracle, and Countdown integration tests."""
import ast
from fractions import Fraction
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

if __package__:
    from . import mcsr_step as m
else:
    import mcsr_step as m


DT = torch.bfloat16
H = 2**-7
BMAX = torch.finfo(DT).max
LEVELS = (8, 128, math.inf)


def words(x):
    return x.view(torch.int16).int() & 65535


def bf16(code):
    return torch.as_tensor(code, dtype=torch.int32).short().view(DT)


def fp32_bits(x):
    return torch.as_tensor(x, dtype=torch.float32).view(torch.int32)


def step(state, d, seed=71, st=1):
    return state.update_([torch.as_tensor(d, dtype=torch.float32)], seed=seed, step=st)


def signed_rank_value(rank):
    word = abs(rank) | (0x8000 if rank < 0 else 0)
    mag = word & 0x7FFF
    exponent, fraction = mag >> 7, mag & 127
    units = fraction * 2**16 if not exponent else (128 + fraction) << (exponent + 15)
    return -units if rank < 0 else units


def position(state):
    e = state._entries[0]
    if e.master is not None:
        return e.master
    s = state._offset(e, e.residual, 0, e.weight.numel())
    return m._decode(e.weight.view(torch.int16).int(), s, e.shift)


def same_state(test, left, right):
    for a, b in zip(left, right):
        for name in a:
            if a[name] is None:
                test.assertIsNone(b[name])
            else:
                av, bv = a[name], b[name]
                if av.dtype == DT:
                    av, bv = words(av), words(bv)
                elif av.dtype == torch.float32:
                    av, bv = av.view(torch.int32), bv.view(torch.int32)
                test.assertTrue(torch.equal(av, bv), name)


class GridTests(unittest.TestCase):
    def test_all_finite_codes_and_neighbor_encodings(self):
        code = torch.arange(65536, dtype=torch.int32)
        code = code[(code & 32767) <= 0x7F7F]
        self.assertEqual(code.numel(), 65280)
        w = bf16(code)
        word = w.view(torch.int16).int()
        rank = m._rank(word)
        self.assertEqual((rank == 0).sum().item(), 2)
        for L in LEVELS:
            state = m.MCSR([w.clone()], L)
            before = state.checkpoint()
            stats = step(state, torch.zeros_like(w, dtype=torch.float32))
            self.assertEqual(stats["bit_changed_count"], 0)
            self.assertEqual(stats["reverse_flip_count"], 0)
            same_state(self, before, state.checkpoint())
            if L == math.inf:
                for mode in ("rtne", "sr"):
                    self.assertTrue(torch.equal(words(state.export_master(0, mode=mode)), code))
                continue
            shift = 16 - int(math.log2(L))
            zero = torch.zeros_like(word)
            self.assertTrue(torch.equal(m._decode(word, zero, shift).view(torch.int32), code << 16))
            low_rank = torch.arange(-0x7F7F, 0x7F7F, dtype=torch.int32)
            low = m._code(low_rank, torch.zeros_like(low_rank, dtype=torch.bool)).short().int()
            high = m._code(low_rank + 1, torch.zeros_like(low_rank, dtype=torch.bool)).short().int()
            for k in (1, L // 2, L - 1):
                a = m._decode(low, torch.full_like(low, k), shift)
                b = m._decode(high, torch.full_like(high, k - L), shift)
                self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))
                expected = (bf16(low).double() * (L - k) + bf16(high).double() * k) / L
                self.assertTrue(torch.equal(a.double(), expected))
                near, away, q = m._mass(a.view(torch.int32), shift)
                self.assertTrue(torch.equal(m._mass_at(low_rank, near, away, q, L),
                                            torch.full_like(low_rank, L - k)))
                self.assertTrue(torch.equal(m._mass_at(low_rank + 1, near, away, q, L),
                                            torch.full_like(low_rank, k)))

    def test_zero_tiny_binade_and_extreme_cases(self):
        w0 = bf16([0, 0x8000, 1, 0x8001, 0x7F7F, 0xFF7F, 0x0080, 0x8080])
        for L in LEVELS:
            w = w0.clone()
            state = m.MCSR([w], L)
            before = state.checkpoint()
            stats = step(state, torch.full(w.shape, -0.0))
            self.assertEqual(stats["bit_changed_count"], 0)
            self.assertEqual(stats["saturated_count"], 0)
            same_state(self, before, state.checkpoint())
            d = torch.tensor([-2**-133, 2**-133, -2**-133, 2**-133,
                              BMAX, -BMAX, -2**-133, 2**-133])
            stats = step(state, d, st=2)
            self.assertEqual(stats["saturated_count"], 2)
            self.assertEqual(stats["reverse_flip_count"], 0)
            self.assertEqual(words(w).tolist(), [0x8001, 1, 0, 0, 0x7F7F, 0xFF7F, 127, 0x807F])
            self.assertTrue(torch.isfinite(w).all())
            self.assertEqual(step(state, torch.zeros_like(d), st=3)["bit_changed_count"], 0)
            sub = m.MCSR([bf16([0, 0x8000])], L)
            quantum = 2**-149 if L == math.inf else 2**-133 / L
            step(sub, [quantum, -quantum])
            self.assertTrue(torch.equal(position(sub), torch.tensor([quantum, -quantum])))

    def test_astra_peak_counterexample_over_many_stream_keys(self):
        n = 1024
        w = torch.ones(n, dtype=DT)
        state = m.MCSR([w], 128)
        src = state.checkpoint()
        src[0]["residual"].fill_(127)
        state.restore_(src)
        for seed in range(128):
            stats = step(state, torch.tensor([0.0, -2**-26]).repeat(n // 2), seed=seed)
            self.assertEqual(stats["bit_changed_count"], 0)
            self.assertEqual(stats["reverse_flip_count"], 0)
            self.assertEqual(stats["swallowed_count"], n // 2)
            self.assertEqual(stats["swallowed_abs_sum"], n // 2 * 2**-26)
        same_state(self, src, state.checkpoint())
        # Use the largest possible fractional random draw and an exact grid point.
        y = fp32_bits(1 + 127 * 2**-14).reshape(1)
        with patch.object(m, "_uniform", return_value=torch.tensor([511], dtype=torch.int32)):
            self.assertTrue(torch.equal(m._fine_bits(y, 9, 0, 0), y))

    def test_fine_bernoulli_exhaustive_draws(self):
        for shift in (9, 13):
            radix = 1 << shift
            coords = torch.arange(radix, dtype=torch.int32)
            for hi in (0x0000, 0x8000, 0x3F7F, 0xBF80, 0x7F7E, 0xFF7E):
                anchor = bf16([hi]).float().view(torch.int32)[0]
                for rem in (0, 1, radix // 2, radix - 1):
                    bits = torch.full_like(coords, anchor + 65536 - radix + rem)
                    with patch.object(m, "_uniform", return_value=coords):
                        rounded = m._fine_bits(bits, shift, 0, 0)
                    self.assertEqual((rounded == anchor + 65536).sum().item(), rem)
                    self.assertEqual((rounded == anchor + 65536 - radix).sum().item(), radix - rem)

    def test_rejection_does_not_shift_other_lanes(self):
        calls = []

        def philox(index, attempt, key):
            calls.append((index.tolist(), attempt.tolist()))
            return torch.tensor([0, 2, 7]) if len(calls) == 1 else torch.tensor([8])

        # 2**32 % 3 == 1, so only lane 20 needs a second draw.
        bounds = torch.tensor([3, 2, 5], dtype=torch.int32)
        with patch.object(m, "_philox_cpu", side_effect=philox):
            got = m._uniform(bounds, 20, 0, bounds=bounds)
        self.assertEqual(got.tolist(), [2, 0, 2])
        self.assertEqual(calls, [([20, 21, 22], [0, 0, 0]), ([20], [1])])
        self.assertEqual(m._philox_cpu(torch.zeros(1, dtype=torch.int64),
                                       torch.zeros(1, dtype=torch.int64), 0).item(), 0x6627E8D5)


class CouplingTests(unittest.TestCase):
    def test_exact_rows_columns_diagonal_direction_and_wasserstein(self):
        # A bitwise-independent oracle for old/new BF16 mass, with unequal
        # binades, sign crossings, completely disjoint support and coarse hits.
        cases = [
            (1.5 + 5 * H / 8, 1.5 + 7 * H / 8, {0x3FC0: 3, 0x3FC1: 5}, {0x3FC0: 1, 0x3FC1: 7}),
            (1 - H / 8, 1 + H / 4, {0x3F7F: 2, 0x3F80: 6}, {0x3F80: 6, 0x3F81: 2}),
            (1 + H / 4, 1 - H / 8, {0x3F80: 6, 0x3F81: 2}, {0x3F7F: 2, 0x3F80: 6}),
            (1 + H / 4, 1 + 4 * H + H / 2, {0x3F80: 6, 0x3F81: 2}, {0x3F84: 4, 0x3F85: 4}),
            (1 - H / 8, 1.0, {0x3F7F: 2, 0x3F80: 6}, {0x3F80: 8}),
            (-2**-134, 2**-134, {-1: 4, 0: 4}, {0: 4, 1: 4}),
        ]
        for L in LEVELS:
            shift = {8: 13, 128: 9, math.inf: 0}[L]
            total = 1 << (16 - shift)
            for x, y, old, new in cases:
                with self.subTest(L=L, x=x, y=y):
                    old = {j: p * total // 8 for j, p in old.items()}
                    new = {j: p * total // 8 for j, p in new.items()}
                    common = {j: min(old.get(j, 0), new.get(j, 0)) for j in old.keys() | new.keys()}
                    rows = {}
                    for i, old_i in old.items():
                        c, e, _, here, keep, rn, ra = (
                            t.item() for t in m._coupling_masses(
                                torch.tensor([i], dtype=torch.int32), fp32_bits([x]), fp32_bits([y]), shift)
                        )
                        self.assertEqual(here, old_i)
                        self.assertEqual(keep, common[i])
                        self.assertEqual(rn, new.get(c, 0) - common.get(c, 0))
                        self.assertEqual(ra, (new.get(e, 0) - common.get(e, 0)) if c != e else 0)
                        row = {j: Fraction(0) for j in new}
                        if keep:
                            row[i] += Fraction(keep, total)
                        if old_i > keep:
                            for j, rem in ((c, rn), (e, ra)):
                                if rem:
                                    row[j] += Fraction((old_i - keep) * rem, total * (rn + ra))
                        self.assertEqual(sum(row.values()), Fraction(old_i, total))
                        for j, weight in row.items():
                            if weight:
                                self.assertGreaterEqual((j - i) * (y - x), 0)
                        rows[i] = row
                    for j, p in new.items():
                        self.assertEqual(sum(row[j] for row in rows.values()), Fraction(p, total))
                        self.assertEqual(rows.get(j, {}).get(j, 0), Fraction(common[j], total))
                    cost = sum(weight * abs(signed_rank_value(j) - signed_rank_value(i))
                               for i, row in rows.items() for j, weight in row.items())
                    self.assertEqual(cost, abs(Fraction(x) - Fraction(y)) * 2**149)

    def test_monte_carlo_marginals_and_transition_probabilities(self):
        n = 120_000
        for L in LEVELS:
            for w0, first, second, old_p, new_p, jump_p in (
                (1.5, 5 * H / 8, H / 4, 5/8, 7/8, 1/4),
                (1 - H / 2, 3 * H / 8, 3 * H / 8, 3/4, 1/4, 1/4),
                (1.5, H / 4, 3 * H, 1/4, 1/4, 1),
            ):
                with self.subTest(L=L, w0=w0, second=second):
                    w = torch.full((n,), w0, dtype=DT)
                    state = m.MCSR([w], L)
                    a = step(state, torch.full((n,), first), seed=17)
                    mid = w.float().clone()
                    b = step(state, torch.full((n,), second), seed=17, st=2)
                    end = w.double()
                    upper = w0 + (H / 2 if w0 < 1 else H)
                    new_upper = 1 + H if w0 < 1 else (w0 + H if second < H else w0 + 4 * H)
                    for actual, expected in ((mid.eq(upper).float().mean().item(), old_p),
                                             (end.eq(new_upper).double().mean().item(), new_p),
                                             (b["step_changed_frac"], jump_p)):
                        self.assertLess(abs(actual - expected), 0.009)
                    self.assertLess(abs(end.mean().item() - (w0 + first + second)), 0.009 * H)
                    self.assertEqual(a["reverse_flip_count"] + b["reverse_flip_count"], 0)
                    if w0 == 1.5 and second < H:
                        retained = end[mid < upper].eq(w0).double().mean().item()
                        self.assertLess(abs(retained - 1/3), 0.016)
                    if w0 < 1:
                        u = torch.rand(n, generator=torch.Generator().manual_seed(171))
                        old_q = torch.where(u < 3/4, 1.0, w0)
                        new_q = torch.where(u < 1/4, 1 + H, 1.0)
                        shared = old_q.ne(new_q).float().mean().item()
                        self.assertLess(abs(shared - 0.5), 0.009)
                        self.assertGreater(shared - b["step_changed_frac"], 0.23)

    def test_finite_subgrid_unbiased_and_closed_loop_boundary(self):
        n = 120_000
        for L in LEVELS:
            state = m.MCSR([torch.full((n,), 1.5, dtype=DT)], L)
            delta = H / (65536 if L == math.inf else L)
            if L != math.inf:
                d = torch.full((n,), delta * 0.375)
                step(state, d)
                x = position(state)
                self.assertLess(abs(x.eq(1.5 + delta).float().mean().item() - 0.375), 0.009)
                self.assertLess(abs(x.double().mean().item() - (1.5 + delta * 0.375)), 0.009 * delta)
            w = torch.full((n,), 1.5, dtype=DT)
            state = m.MCSR([w], L)
            step(state, torch.full((n,), H / 2))
            old = w.clone()
            stats = step(state, (old > 1.5).float() * (H / 2), st=2)
            self.assertTrue(torch.equal(w, old))
            self.assertEqual(stats["bit_changed_count"], 0)
            self.assertLess(abs((position(state).double().mean().item() - 1.5) / H - 0.75), 0.009)


class StateTests(unittest.TestCase):
    def test_actual_int4_bytes_and_invalid_tail(self):
        for n in (1, 3, 17, 1025):
            w = torch.ones(n, dtype=DT)
            state = m.MCSR([w], 8, chunk_size=17)
            s = torch.arange(n, dtype=torch.int32) % 15 - 7
            packed = torch.empty((n + 1) // 2, dtype=torch.uint8)
            m._pack_into(packed, s)
            self.assertTrue(torch.equal(m._unpack(packed, n), s))
            self.assertEqual(packed.untyped_storage().nbytes(), (n + 1) // 2)
            self.assertEqual(packed[-1].item() >> 4, 0)
            self.assertEqual(state.state_bytes, 2 * n + (n + 1) // 2)
            src = state.checkpoint()
            src[0]["residual"].copy_(packed)
            state.restore_(src)
            self.assertEqual(step(state, torch.zeros(n))["bit_changed_count"], 0)
            same_state(self, src, state.checkpoint())
            packed[-1] |= 0xF0
            src[0]["residual"].copy_(packed)
            with self.assertRaises(ValueError):
                state.restore_(src)

    def test_chunk_restart_and_tied_weights(self):
        gen = torch.Generator().manual_seed(624)
        w0 = torch.randn(217, generator=gen).bfloat16()
        w1 = torch.randn(43, generator=gen).bfloat16()
        deltas = [[torch.randn(w.shape, generator=gen) * 0.002 for w in (w0, w1)] for _ in range(3)]
        for L in LEVELS:
            with self.subTest(L=L):
                w = w0.clone()
                base = m.MCSR([w, w.detach(), w1.clone()], L)
                self.assertEqual(len(base.weights), 2)
                s1 = base.update_(deltas[0], seed=9, step=1)
                src = base.checkpoint()
                s2 = base.update_(deltas[1], seed=9, step=2)
                s3 = base.update_(deltas[2], seed=9, step=3)
                for size in (1, 7, 32, 5_000):
                    obj = m.MCSR([w0.clone(), w1.clone()], L, chunk_size=size)
                    self.assertEqual(obj.update_(deltas[0], seed=9, step=1), s1)
                    same_state(self, obj.checkpoint(), src)
                    obj.restore_(src)
                    self.assertEqual(obj.update_(deltas[1], seed=9, step=2), s2)
                    self.assertEqual(obj.update_(deltas[2], seed=9, step=3), s3)
                    same_state(self, base.checkpoint(), obj.checkpoint())
                    if L == math.inf:
                        for mode in ("rtne", "sr"):
                            self.assertTrue(torch.equal(words(base.export_master(0, mode=mode, seed=9)),
                                                        words(obj.export_master(0, mode=mode, seed=9))))

    def test_incompatible_inputs_fail_before_any_write(self):
        for L in LEVELS:
            a, b = [torch.ones(2, 3, dtype=DT) for _ in range(2)]
            state = m.MCSR([a, b], L, chunk_size=3)
            src = state.checkpoint()
            bad_grads = [b.flatten(), b.float().T, b, b.double(),
                         torch.full(b.shape, math.nan), torch.full(b.shape, math.inf)]
            for bad in bad_grads:
                with self.assertRaises(ValueError):
                    state.update_([torch.ones_like(a), bad], seed=0, step=1)
                same_state(self, src, state.checkpoint())
            bad = state.checkpoint()
            if L == 8:
                bad[-1]["residual"][-1] = 8
            elif L == 128:
                bad[-1]["residual"][-1] = -128
            else:
                bad[-1]["master"][-1, -1] = 2.0
            with self.assertRaises(ValueError):
                state.restore_(bad)
            same_state(self, src, state.checkpoint())
        with self.assertRaises(ValueError):
            m.MCSR([a.flatten()[:3], a.flatten()[1:4]], 8)

    def test_fp32_master_matches_fp32_add_and_exports(self):
        rng = torch.Generator().manual_seed(64)
        for L in (math.inf,):
            w = torch.randn(1701, generator=rng).bfloat16()
            state = m.MCSR([w], L)
            ref = w.float()
            for st in range(1, 7):
                d = torch.randn(w.shape, generator=rng) * 0.007
                ref.add_(d)
                self.assertEqual(step(state, d, st=st)["reverse_flip_count"], 0)
                self.assertTrue(torch.equal(position(state).view(torch.int32), ref.view(torch.int32)))
            native = state.checkpoint()
            self.assertTrue(torch.equal(words(state.export_master(0)), words(ref.bfloat16())))
            self.assertFalse(torch.equal(words(state.export_master(0, mode="sr", seed=10)),
                                         words(state.export_master(0, mode="sr", seed=11))))
            same_state(self, native, state.checkpoint())


class CountdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        probe = Path(__file__).resolve().parents[1] / "experiments"
        if not probe.is_dir():
            probe = Path(__file__).resolve().parents[1] / "probe_top2_closedloop"
        sys.path.insert(0, str(probe))
        import run_countdown_mcsr_v22 as run
        import verify_mcsr_smoke as smoke
        cls.run22, cls.smoke = run, smoke
        cls.original = (probe / "run_countdown_rlvr_v2.py").read_text()

    def test_frozen_protocol_and_ast_verifier(self):
        train, dev, kl, test = self.run22.locked_countdown_splits()
        self.assertEqual(tuple(map(len, (train, dev, kl, test))), (240, 40, 12, 200))
        all_rows = train + dev + kl + test
        self.assertEqual(len({tuple(sorted(i["nums"])) for i in all_rows}), 492)
        for i in all_rows:
            self.assertTrue(self.run22.safe_eval_countdown(i["ref_expr"], i["nums"], i["target"]))
        out, rate = self.run22.grade_countdown_batch(
            [r"\boxed{(2 + 3) * 4} \boxed{(2 + 3) * 4=19}",
             r"\boxed{(2 + 3) * 4} \boxed{incomplete",
             r"\boxed{import os (2 + 3) * 4}"], [{"nums": [2, 3, 4], "target": 20}] * 3)
        self.assertEqual(out, [False, False, False])
        self.assertEqual(rate, 2 / 3)
        a = ast.parse(self.original)
        b = ast.parse(Path(self.run22.__file__).read_text())
        get = lambda tree: {f.name: ast.dump(f) for f in tree.body if isinstance(f, ast.FunctionDef)}
        for name in ("generate_disjoint_countdown_splits", "format_countdown_prompt",
                     "extract_last_boxed", "safe_eval_countdown", "matched_generate_batch",
                     "cache_anchor_kl", "compute_kl", "evaluate_countdown"):
            self.assertEqual(get(a)[name], get(b)[name])

    def test_legacy_update_parity(self):
        legacy = next(f for f in ast.parse(self.original).body
                      if isinstance(f, ast.FunctionDef) and f.name == "apply_step")
        scope = vars(self.run22).copy()
        exec(compile(ast.Module(body=[legacy], type_ignores=[]), "<original-v2.1>", "exec"), scope)
        legacy = scope["apply_step"]
        dev = torch.device("cpu")
        for arm in self.run22.BASE_ARMS[1:]:
            torch.manual_seed(61)
            a, b = self.smoke.TinyLM(dev), self.smoke.TinyLM(dev)
            b.load_state_dict(a.state_dict())
            ma = [p.detach().float().clone() for p in a.parameters()] if arm.startswith("M_") else None
            mb = [x.clone() for x in ma] if ma is not None else None
            for st in (1, 2):
                args = (0.05, 42, st, 11, 19)
                prev = legacy(arm, a, self.smoke.MIXED, ma, *args)
                new = self.run22.apply_step(arm, b, self.smoke.MIXED, mb, *args)
                self.assertEqual(prev, new[:3])
                for i, (x, y) in enumerate(zip(a.parameters(), b.parameters())):
                    self.assertTrue(torch.equal(words(x), words(y)))
                    if ma is not None:
                        self.assertTrue(torch.equal(ma[i], mb[i]))

    def test_all_arms_backward_and_saved_state(self):
        reports = self.smoke.verify_arms(torch.device("cpu"))
        self.assertEqual({r["arm"]: r["state_bytes"] for r in reports},
                         {"MCSR_int4": 2168, "MCSR_int8": 2601, "M_coupled": 5202,
                          "S_norm_fp32": 1798, "MCSR_int4_norm_fp32": 2224})
        model = self.smoke.TinyLM(torch.device("cpu"))
        ps = list(model.parameters())
        state = self.run22.create_arm_state("MCSR_int4_norm_fp32", ps, model)
        self.run22.apply_step("MCSR_int4_norm_fp32", model, self.smoke.MIXED, None,
                              0.05, 42, 1, rows=11, state=state)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            self.run22.save_final_checkpoint(path, model, ps, state, None, {"seed": 42}, "fixture", 1)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        same_state(self, ckpt["parameters"], state.checkpoint())
        self.assertEqual(ckpt["step"], 1)
        self.assertEqual(ckpt["name_to_index"]["head.weight"],
                         ckpt["name_to_index"]["model.embedding.weight"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
