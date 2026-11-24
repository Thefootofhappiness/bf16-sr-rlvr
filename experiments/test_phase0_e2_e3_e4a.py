#!/usr/bin/env python3
"""Local CPU tests for E2 coupled SGD, E3 AdamW exports, and E4a diagnostics."""

import copy
import json
import math
from fractions import Fraction
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import run_matched_ood_v22 as runner
from test_matched_ood_v22 import RECORDS, bits, tiny_model


def frozen_gsm8k_grade(preds, golds):
    """Pre-Phase-0 decisions with the original last-opener and marker rules."""
    out = []
    for text, gold in zip(preds, golds):
        try:
            gold_value = Fraction(gold.replace(",", "").strip())
            candidate = None
            offset = text.rfind(r"\boxed{")
            if offset >= 0:
                start = offset + len(r"\boxed{")
                depth = 1
                for end in range(start, len(text)):
                    depth += (text[end] == "{") - (text[end] == "}")
                    if depth == 0:
                        candidate = text[start:end].strip()
                        break
            if candidate is None:
                m = re.search(r"(?:####|final answer\s*(?:is|:)|the answer is)\s*([^\n\.]+)", text, re.IGNORECASE)
                candidate = m.group(1).strip() if m else None
            parsed = runner.parse_gsm8k_number(candidate)
            out.append(parsed is not None and parsed == gold_value)
        except Exception:
            out.append(False)
    return out


def two_point_law(x):
    """Independent oracle based on numerical BF16 neighbors and mass."""
    nearest = x.bfloat16()
    lo = torch.where(nearest.float() > x, torch.nextafter(nearest, torch.full_like(nearest, -math.inf)), nearest)
    hi = torch.nextafter(lo, torch.full_like(lo, math.inf))
    q = (x.double() - lo.double()) / (hi.double() - lo.double())
    return lo, hi, q


class PhaseZeroTests(unittest.TestCase):
    def test_adamw_master_and_moments_bitwise_and_idle(self):
        base = tiny_model(torch.device("cpu"), tied=True)
        masters0 = [p.detach().float() for p in runner.get_unique_params(base)]
        runs = {}
        for arm in runner.ADAM_ARMS:
            with self.subTest(arm=arm):
                model = copy.deepcopy(base)
                masters = copy.deepcopy(masters0)
                state = runner.create_adam_state(masters, 0.01)
                result = runner.apply_matched_step(
                    arm, model, RECORDS, masters, 0.00137, 42, 1,
                    rows=7, chunk=29, adam_state=state, master_telemetry=True,
                )
                self.assertTrue(result[2])
                self.assertEqual(state["t"], 1)
                n_params = sum(p.numel() for p in model.parameters())
                changes = sum(torch.count_nonzero(bits(p) != bits(p0)).item()
                              for p, p0 in zip(model.parameters(), base.parameters(), strict=True))
                self.assertEqual(result[1], changes / n_params)
                self.assertEqual(result[-1]["coo_bytes_raw"], 6 * changes)
                runs[arm] = (model, masters, state)
                idle = [dict(record, advs=[0.0, 0.0]) for record in RECORDS]
                snapshot = [bits(m).clone() for m in masters]
                served = [bits(p).clone() for p in model.parameters()]
                idle_result = runner.apply_matched_step(arm, model, idle, masters, 0.00137, 42, 2,
                                                        adam_state=state, master_telemetry=True)
                self.assertEqual(idle_result[:3], (0.0, 0.0, False))
                self.assertEqual(state["t"], 1)
                self.assertTrue(all(torch.equal(a, bits(b)) for a, b in zip(snapshot, masters, strict=True)))
                self.assertTrue(all(torch.equal(a, bits(b)) for a, b in zip(served, model.parameters(), strict=True)))

        rtne, ref_masters, ref_state = runs["AdamW_fp32"]
        for arm in runner.ADAM_ARMS[1:]:
            _, masters, state = runs[arm]
            max_diff = max((m - r).abs().max().item() for m, r in zip(masters, ref_masters, strict=True))
            self.assertEqual(max_diff, 0.0)
            for actual, expected in ((masters, ref_masters), (state["m_cpu"], ref_state["m_cpu"]),
                                     (state["v_cpu"], ref_state["v_cpu"])):
                self.assertTrue(all(torch.equal(bits(a), bits(b)) for a, b in zip(actual, expected, strict=True)))

        replicas = runs["AdamW_faircast_sr"][0].parameters()
        sr_diff = 0
        for j, (observed, master, p_rtne) in enumerate(zip(replicas, ref_masters, rtne.parameters(), strict=True)):
            # For these nonzero gradients, Fair Cast uses the same per-parameter RNG as M_faircast.
            gen = torch.Generator(device="cpu").manual_seed(42 * 1000003 + 1009 + j)
            expected = torch.empty_like(observed)
            runner.sr_into(master, expected, gen, chunk=29)
            self.assertTrue(torch.equal(bits(observed), bits(expected)))
            self.assertTrue(torch.equal(bits(p_rtne), bits(master.bfloat16())))
            sr_diff += torch.count_nonzero(bits(observed) != bits(p_rtne)).item()
        self.assertGreater(sr_diff, 0)

        coupled, masters, state = runs["AdamW_coupled_sr"]
        assert any((m != m.bfloat16().float()).any() for m in masters)
        old = [bits(p).clone() for p in coupled.parameters()]
        result = runner.apply_matched_step("AdamW_coupled_sr", coupled, RECORDS, masters, 0.0, 42, 2,
                                           rows=7, chunk=29, adam_state=state, master_telemetry=True)
        self.assertTrue(result[2])
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[-1], {"coo_bytes_raw": 0, "reverse_flip_count": 0})
        self.assertTrue(all(torch.equal(a, bits(b)) for a, b in zip(old, coupled.parameters(), strict=True)))

    def test_coupled_5000_transitions_match_tv_and_marginals(self):
        gen = torch.Generator(device="cpu").manual_seed(71041)
        n = 5000
        boundary = torch.exp2(torch.randint(-9, 10, (n,), generator=gen).float())
        sign = 2 * torch.randint(2, (n,), generator=gen) - 1
        side = torch.arange(n) % 2 == 0
        x = sign * boundary * (1 - (0.05 + 0.90 * torch.rand(n, generator=gen)) / 256)
        # Half cross a binade boundary. Half stay on the same adjacent BF16 support.
        y = sign * boundary * (1 + torch.where(side, 2, -1)
                               * (0.05 + 0.90 * torch.rand(n, generator=gen)) / 256)
        self.assertTrue(torch.equal(x + (y - x), y))
        old = torch.empty_like(x, dtype=torch.bfloat16)
        runner.sr_into(x, old, gen)
        lo_x, hi_x, q_x = two_point_law(x)
        lo_y, hi_y, q_y = two_point_law(y)
        overlap = torch.zeros_like(q_x)
        for old_value, old_mass in ((lo_x, 1 - q_x), (hi_x, q_x)):
            for new_value, new_mass in ((lo_y, 1 - q_y), (hi_y, q_y)):
                overlap += torch.where(old_value == new_value, torch.minimum(old_mass, new_mass), 0)
        tv = 1 - overlap

        serving = old.clone()
        sgd = runner.create_mcsr_state("M_coupled", [serving], chunk_size=311)
        self.assertEqual(sgd.state_bytes, 6 * n)
        self.assertEqual(sgd.checkpoint()[0]["master"].device.type, "cpu")
        sgd.restore_([{"weight": old, "master": x, "residual": None}])
        quiet = sgd.update_([torch.zeros_like(x)], seed=13, step=1)
        self.assertEqual((quiet["bit_changed_count"], quiet["reverse_flip_count"]), (0, 0))
        self.assertTrue(torch.equal(bits(serving), bits(old)))
        self.assertTrue(torch.equal(bits(sgd.checkpoint()[0]["master"]), bits(x)))
        still = runner.coupled_master_export(x, old, x, 13, 1, 0, 0)
        self.assertTrue(torch.equal(bits(still), bits(old)))

        stats = sgd.update_([y - x], seed=13, step=2)
        adam = runner.coupled_master_export(x, old, y, 13, 2, 0, 0)
        self.assertEqual(stats["reverse_flip_count"], 0)
        self.assertTrue(torch.equal(bits(sgd.checkpoint()[0]["master"]), bits(y)))
        self.assertTrue(torch.equal(bits(adam), bits(serving)))
        for arm, new in (("M_coupled", serving), ("AdamW_coupled_sr", adam)):
            with self.subTest(arm=arm):
                reverse = runner.count_reverse_flips(x, y, old, new).item()
                self.assertEqual(reverse, 0)
                self.assertTrue(((new == lo_y) | (new == hi_y)).all())
                flipped = (bits(new) != bits(old)).double()
                upper = (new == hi_y).double()
                for group in (torch.ones_like(side), side, ~side, sign < 0, sign > 0):
                    for draws, prob in ((flipped, tv), (upper, q_y)):
                        error = (draws[group] - prob[group]).sum().item()
                        sigma = (prob[group] * (1 - prob[group])).sum().sqrt().item()
                        self.assertLess(abs(error), 6 * sigma)
                print(f"\n{arm}: {n} transitions, opposite={reverse}, "
                      f"flips={int(flipped.sum())}, expected_TV={tv.mean():.8f}, "
                      f"flip_rate={flipped.mean():.8f}, mean_upper_error={(upper - q_y).mean():+.8f}")

    def test_coupled_sgd_uses_served_model_and_logs_counts(self):
        model = tiny_model(torch.device("cpu"), tied=False)
        ps = runner.get_unique_params(model)
        state = runner.create_mcsr_state("M_coupled", ps, chunk_size=31)
        before = state.checkpoint()
        norm, rate, did, _, _, stats = runner.apply_matched_step(
            "M_coupled", model, RECORDS, None, 0.1, 42, 1, rows=7, chunk=29, state=state,
        )
        n = sum(p.numel() for p in ps)
        changed = sum(torch.count_nonzero(bits(p) != bits(entry["weight"])).item()
                      for p, entry in zip(ps, before, strict=True))
        self.assertTrue(did and norm > 0 and changed > 0)
        self.assertEqual(rate, changed / n)
        self.assertEqual(stats["coo_bytes_raw"], 6 * changed)
        self.assertEqual(stats["reverse_flip_count"], 0)

    def test_gsm8k_edge_cases_through_evaluation_and_jsonl(self):
        edge_cases = [
            ("The final answer is: 18", "18", "parse_fail", "explicit_marker", None),
            (r"\boxed{18} But trailing \boxed{", "18", "no_candidate", "none", None),
            (r"The answer is 18. \boxed{}", "18", "parse_fail", "boxed", None),
            ("#### 17\nFinal answer: 18", "18", "wrong_value", "explicit_marker", "17"),
            ("The answer is 3.5", "3", "correct", "explicit_marker", "3"),
        ]
        preds = [rec[0] for rec in edge_cases]
        golds = [rec[1] for rec in edge_cases]
        self.assertEqual(runner.grade_gsm8k_batch(preds, golds), [False, False, False, False, True])
        self.assertEqual(runner.grade_gsm8k_batch(preds, golds), frozen_gsm8k_grade(preds, golds))
        extra = [r"\boxed{\frac{18}{2}}", r"\boxed{1} #### 18 \boxed{", "Answer 18",
                 "Final answer:", "The answer is 3.5", r"\boxed{1,234}", r"\boxed{1e3}"]
        extra_golds = ["9", "18", "18", "18", "7/2", "1234", "1000", "bad"]
        self.assertEqual(runner.grade_gsm8k_batch(extra, extra_golds), frozen_gsm8k_grade(extra, extra_golds))
        items = [{"_idx": i + 10, "problem": str(i), "answer": g} for i, g in enumerate(golds)]

        def mock_generate(*args, n_samples, **kwargs):
            return [{"texts": [p] * n_samples, "comp_ids": [[4, 2]] * n_samples,
                     "truncs": [i == 1] * n_samples} for i, p in enumerate(preds)]

        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "matched_generate_batch", mock_generate):
            path = Path(temp) / "gsm8k_ood_completions.jsonl"
            diag = {}
            saved = runner.evaluate_suite(None, None, items, runner.grade_gsm8k_batch, "cpu", 50, 2, 77,
                                          completions_path=path, grader_diag=diag)
            original = runner.evaluate_suite(None, None, items, runner.grade_gsm8k_batch, "cpu", 50, 2, 77)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        saved.pop("eval_sec")
        original.pop("eval_sec")
        self.assertEqual(saved, original)
        self.assertEqual(len(rows), 15)
        self.assertEqual(diag["greedy"]["failure_categories"],
                         {"no_candidate": 1, "parse_fail": 2, "wrong_value": 1, "correct": 1})
        self.assertEqual(diag["sample_k"]["truncated"], 2)
        for i, row in enumerate(rows[:5]):
            self.assertEqual((row["failure_category"], row["selected_branch"], row["parsed_value"]), edge_cases[i][2:])
            self.assertEqual((row["item_idx"], row["mode"], row["n_tokens"]), (10 + i, "greedy", 2))
        self.assertEqual([b["closed"] for b in rows[1]["all_boxed"]], [True, False])
        self.assertEqual(rows[2]["all_boxed"][0]["value"], "")
        self.assertEqual([m["candidate"] for m in rows[3]["explicit_markers"]], ["17", "18"])
        self.assertEqual([rec["sample_idx"] for rec in rows[5:]], [0, 1] * 5)
        self.assertTrue(all(rec["mode"] == "sample_k" for rec in rows[5:]))

    def test_math_jsonl_uses_actual_math_parser_trace(self):
        items = [{"_idx": 16, "problem": "Compute 3.5", "answer": "3.5"}]
        outs = [{"texts": ["The answer is 3.5"], "comp_ids": [[3, 5, 2]], "truncs": [False]}]

        def grade(preds, golds, *, diagnostics=None):
            if diagnostics is not None:
                diagnostics.append({"grade_status": "graded", "prediction_parse": ["3.5"]})
            return [True]

        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "matched_generate_batch", return_value=outs):
            path = Path(temp) / "math500_completions.jsonl"
            runner.evaluate_suite(None, None, items, grade, "cpu", 50, 0, 77, completions_path=path)
            line = json.loads(path.read_text())
        self.assertEqual((line["selected_branch"], line["parsed_value"], line["grade_ok"],
                          line["failure_category"]), ("math_verify", ["3.5"], True, "correct"))

    def test_sticking_and_exact_master_reference_token_logps(self):
        serving = tiny_model(torch.device("cpu"), tied=True).eval()
        params = runner.get_unique_params(serving)
        initial = [p.detach().clone() for p in params]
        masters = [p.detach().float() for p in params]
        masters[0].flatten()[0].add_(1e-5)
        p1 = params[0].view(-1)[1:2]
        with torch.no_grad():
            p1.copy_(torch.nextafter(p1, torch.full_like(p1, math.inf)))
        masters[0].flatten()[1] = p1.detach().float()[0]
        self.assertEqual(runner.sticking_fraction(params, masters, initial),
                         1 / sum(p.numel() for p in params))
        shadow = copy.deepcopy(serving).float()
        runner.load_fp32_master(shadow, masters)
        self.assertTrue(all(torch.equal(bits(a), bits(b)) for a, b in zip(shadow.parameters(), masters, strict=True)))
        ref = [{"prompt_ids": [1, 6], "comp_ids": [[5, 7, 2], []]}]
        actual = runner.reference_token_logps(shadow, ref, torch.device("cpu"))
        expected = F.log_softmax(shadow(input_ids=torch.tensor([[1, 6, 5, 7, 2]])).logits[0, 1:-1], dim=-1)
        torch.testing.assert_close(actual, expected[torch.arange(3), [5, 7, 2]])
        self.assertEqual(runner.reference_token_logps(serving, ref, torch.device("cpu")).numel(), 3)

    def test_final_checkpoint_preserves_packed_state_and_aliases(self):
        model = tiny_model(torch.device("cpu"), tied=True)
        ps = runner.get_unique_params(model)
        state = runner.create_mcsr_state("MCSR_int4", ps, chunk_size=29)
        runner.apply_matched_step("MCSR_int4", model, RECORDS, None, 0.1, 42, 1, rows=7, state=state)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "weights.pt"
            runner.save_final_checkpoint(path, model, ps, state, None, {"arm": "MCSR_int4"}, "test", 1)
            saved = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(saved["name_to_index"]["model.embed_tokens.weight"], saved["name_to_index"]["lm_head.weight"])
        self.assertTrue(any(p.any() for p in saved["S_pack"]))
        for actual, residual, expected in zip(saved["parameters"], saved["S_pack"], state.checkpoint(), strict=True):
            self.assertTrue(torch.equal(bits(actual["weight"]), bits(expected["weight"])))
            self.assertTrue(torch.equal(residual, expected["residual"]))
            self.assertEqual(residual.dtype, torch.uint8)
            self.assertEqual(residual.untyped_storage().nbytes(), (actual["weight"].numel() + 1) // 2)

    def test_cli_gsm_only_and_exact_master_end_outputs(self):
        class CPUOnlyTorch:
            device = staticmethod(lambda _: torch.device("cpu"))
            cuda = SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda: 0,
                                   empty_cache=lambda: None, reset_peak_memory_stats=lambda: None)

            def __getattr__(self, name):
                return getattr(torch, name)

        tokenizer = SimpleNamespace(padding_side=None, pad_token_id=0, encode=lambda *a, **k: [1, 6, 2])
        generation_calls = []
        math_calls = []

        def generate(model, tok, prompts, device, max_new_tokens, n_samples, temperature, seed, batch_size):
            generation_calls.append((next(model.parameters()).dtype, len(prompts), n_samples, seed))
            return [{"prompt_ids": [1, 6], "texts": ["#### 18" if k % 2 == 0 else "#### 17" for k in range(n_samples)],
                     "comp_ids": [[4 + k, 2] for k in range(n_samples)], "truncs": [False] * n_samples}
                    for _ in prompts]

        def math_grade(preds, golds, *, diagnostics=None):
            math_calls.append(len(preds))
            if diagnostics is not None:
                diagnostics.extend({"grade_status": "graded", "prediction_parse": [p.split()[-1]]} for p in preds)
            return [p == "#### 18" for p in preds]

        def pretrained(model_path, torch_dtype, attn_implementation):
            return tiny_model(torch.device("cpu"), tied=True).to(dtype=torch_dtype)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            math_row = {"problem": "Compute 18", "answer": "18"}
            gsm_row = {"question": "Compute 18", "answer": "#### 18"}
            (root / "math.jsonl").write_text((json.dumps(math_row) + "\n") * 492)
            (root / "gsm.jsonl").write_text((json.dumps(gsm_row) + "\n") * 200)
            (root / "splits.json").write_text(json.dumps({"dev_ids": list(range(292)),
                                                         "test_ids": list(range(292, 492))}))
            (root / "wt.txt").write_text("A small CPU fixture")
            out = root / "result.json"
            dumps = root / "completions"
            ckpt = root / "final.pt"
            argv = ["runner", "--arm", "AdamW_coupled_sr", "--model", "tiny-fixture",
                    "--math-data", str(root / "math.jsonl"), "--math-splits", str(root / "splits.json"),
                    "--gsm8k-data", str(root / "gsm.jsonl"), "--wikitext-data", str(root / "wt.txt"),
                    "--pydeps", str(root), "--steps", "1", "--batch-prompts", "1", "--G", "2",
                    "--eta", "0.00137", "--eval-n", "2", "--skip-math-eval", "--eval-master-end",
                    "--save-eval-completions", str(dumps), "--save-final-ckpt", str(ckpt), "--out", str(out)]
            def run(args):
                with (patch.object(runner, "torch", CPUOnlyTorch()),
                      patch.object(runner.sys, "argv", args),
                      patch.object(runner, "load_math_grader", return_value=(math_grade, [0])),
                      patch.object(runner.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                      patch.object(runner.AutoModelForCausalLM, "from_pretrained", side_effect=pretrained),
                      patch.object(runner, "matched_generate_batch", side_effect=generate)):
                    runner.main()

            run(argv)
            result = json.loads(out.read_text())
            self.assertTrue(ckpt.is_file())
            self.assertFalse(list(dumps.rglob("math500_completions.jsonl")))
            self.assertEqual(sorted(len(p.read_text().splitlines()) for p in dumps.rglob("gsm8k_ood_completions.jsonl")),
                             [600, 1800])
            coupled_math_calls = list(math_calls)
            coupled_generation_calls = list(generation_calls)
            faircast = ["AdamW_faircast_sr" if arg == "AdamW_coupled_sr" else arg
                        for arg in argv if arg != "--skip-math-eval"]
            run(faircast)
            faircast_result = json.loads(out.read_text())
            saved = torch.load(ckpt, map_location="cpu", weights_only=True)
            master = torch.cat([entry["master"].flatten() for entry in saved["parameters"]])
            q = (bits(master) & 65535).double() / 65536
            audit = faircast_result["master_audit"]
            self.assertAlmostEqual(audit["theoretical_2p1mp"], (2 * q * (1 - q)).mean().item())
            self.assertEqual(len(audit["empirical_adj_flips_4pairs"]), 4)
            self.assertGreater(audit["empirical_adj_flip_mean"], 0.0)
            self.assertEqual(faircast_result["master_end_eval"]["math500_test"]["n_items"], 200)
            self.assertEqual(faircast_result["master_end_eval"]["math500_test"]["n_samples"], 8)
            self.assertEqual(len((dumps / "master_end_fp32" / "math500_completions.jsonl").read_text().splitlines()),
                             1800)

        self.assertEqual(coupled_math_calls, [2])
        self.assertEqual(result["effective_steps"], 1)
        self.assertIsNone(result["math500_test"])
        self.assertIsNone(result["checkpoints"][0]["dev40_greedy"])
        self.assertEqual(result["step_logs"][0]["reverse_flip_count"], 0)
        self.assertIn("sticking_frac", result["step_logs"][0])
        self.assertGreater(result["checkpoints"][0]["hidden_lag_mean_abs_dlogp"], 0)
        self.assertEqual(result["gsm8k_grader_diag"]["sample_k"]["n"], 400)
        self.assertIsNone(result["master_end_eval"]["math500_test"])
        self.assertEqual(result["master_end_eval"]["gsm8k_ood"]["n_samples"], 8)
        eval_calls = [call[:3] for call in coupled_generation_calls if call[1] == 200]
        self.assertEqual(eval_calls, [(torch.bfloat16, 200, 1), (torch.bfloat16, 200, 2),
                                      (torch.float32, 200, 1), (torch.float32, 200, 8)])


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
