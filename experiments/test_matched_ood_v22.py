#!/usr/bin/env python3
"""Offline matched-step tests. CUDA cases also run when CUDA is available."""

import copy
import math
import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import run_matched_ood_v2 as v21
import run_matched_ood_v22 as v22


DEVICES = (torch.device("cpu"),)
if torch.cuda.is_available():
    DEVICES += (torch.device("cuda:0"),)

RECORDS = [
    {"prompt_ids": [3, 8, 7], "comp_ids": [[4, 5, 2], [5, 9, 2]], "advs": [.5, -.5]},
    {"prompt_ids": [2, 6], "comp_ids": [[8, 3], [9, 7, 1]], "advs": [-.5, .5]},
]


def tiny_model(device, tied):
    config = Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attention_dropout=0., tie_word_embeddings=tied, bos_token_id=1,
        eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(311)
    return Qwen2ForCausalLM(config).to(device=device, dtype=torch.bfloat16)


def bits(tensor):
    dtype = torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32
    return tensor.detach().cpu().view(dtype)


class MatchedOODTests(unittest.TestCase):
    def test_mcsr_mixed_advantage_and_physical_state(self):
        for device in DEVICES:
            for arm, level, dtype, weights_per_byte in (
                ("MCSR_int4", 8, torch.uint8, 2),
                ("MCSR_int8", 128, torch.int8, 1),
            ):
                with self.subTest(device=device, arm=arm):
                    model = tiny_model(device, tied=True)
                    params = v22.get_unique_params(model)
                    n_params = sum(p.numel() for p in params)
                    self.assertTrue(all(p.numel() % 2 == 0 for p in params))
                    state = v22.create_mcsr_state(arm, params, chunk_size=31)
                    self.assertEqual((state.L, state.chunk_size), (level, 31))
                    expected_size = 2 * n_params + n_params // weights_per_byte
                    self.assertEqual(state.state_bytes, expected_size)
                    self.assertEqual(state.state_bytes_on(device), expected_size)
                    old = [bits(p).clone() for p in params]
                    for step in (1, 2):
                        raw_norm, frac, did, _, update_sec, stats = v22.apply_matched_step(
                            arm, model, RECORDS, None, .1, 64, step, rows=7, chunk=29, state=state
                        )
                        self.assertTrue(did)
                        self.assertTrue(math.isfinite(raw_norm) and raw_norm > 0)
                        self.assertGreater(update_sec, 0.)
                        self.assertEqual(stats["reverse_flip_count"], 0)
                        self.assertEqual(stats["saturated_count"], 0)
                        self.assertGreaterEqual(stats["swallowed_count"], 0)
                        new = [bits(p).clone() for p in params]
                        changed = sum(int((a != b).sum()) for a, b in zip(old, new, strict=True))
                        self.assertGreater(changed, 0)
                        self.assertEqual(frac, changed / n_params)
                        self.assertTrue(all(torch.isfinite(p).all() and p.grad is None for p in params))
                        old = new

                    packed = state.checkpoint()
                    for param, entry in zip(params, packed, strict=True):
                        residual = entry["residual"]
                        self.assertIsNone(entry["master"])
                        self.assertEqual(residual.dtype, dtype)
                        self.assertEqual(residual.untyped_storage().nbytes(),
                                         param.numel() // weights_per_byte)
                    self.assertTrue(any(entry["residual"].any() for entry in packed))
                    idle = [dict(record, advs=[0., 0.]) for record in RECORDS]
                    result = v22.apply_matched_step(
                        arm, model, idle, None, .1, 64, 3, rows=7, chunk=29, state=state
                    )
                    self.assertEqual(result[:3], (0., 0., False))
                    self.assertEqual(result[4:], (0., {}))
                    for before, after in zip(packed, state.checkpoint(), strict=True):
                        self.assertTrue(torch.equal(bits(before["weight"]), bits(after["weight"])))
                        self.assertTrue(torch.equal(before["residual"], after["residual"]))

    def test_legacy_training_arms_bitwise(self):
        for device in DEVICES:
            for tied in (False, True):
                base = tiny_model(device, tied=tied)
                for arm in ("S_full_sr", "M_faircast", "M_rtne", "R_pure_bf16"):
                    with self.subTest(device=device, tied=tied, arm=arm):
                        locked, extended = copy.deepcopy(base), copy.deepcopy(base)
                        old_master = ([p.detach().cpu().float() for p in locked.parameters()]
                                      if arm in ("M_faircast", "M_rtne") else None)
                        new_master = copy.deepcopy(old_master)
                        sync = (patch.object(v21.torch.cuda, "synchronize") if device.type == "cpu"
                                else nullcontext())
                        with sync:
                            expected = v21.apply_matched_step(
                                arm, locked, RECORDS, old_master, .1, 64, 1, rows=7, chunk=29
                            )
                        actual = v22.apply_matched_step(
                            arm, extended, RECORDS, new_master, .1, 64, 1, rows=7, chunk=29
                        )
                        self.assertTrue(expected[2])
                        self.assertEqual(actual[:3], expected[:3])
                        for a, b in zip(locked.parameters(), extended.parameters(), strict=True):
                            self.assertTrue(torch.equal(bits(a), bits(b)))
                        if old_master is not None:
                            for a, b in zip(old_master, new_master, strict=True):
                                self.assertTrue(torch.equal(bits(a), bits(b)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unittest.main(verbosity=2)
