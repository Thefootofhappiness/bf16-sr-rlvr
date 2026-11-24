#!/usr/bin/env python3
"""Small live backward/update smoke. Run under either target CUDA PyTorch build.

CUDA_VISIBLE_DEVICES=<allocated 4090> python verify_mcsr_smoke.py
CPU mode runs the same integration without Transformers model assets or CUDA.
"""
import argparse
import json
import time
from types import SimpleNamespace

import torch

from run_countdown_mcsr_v22 import (
    MCSR_ARMS, NORM_ARMS, apply_step, create_arm_state, locked_countdown_splits,
)
from mcsr_step import DEFAULT_CHUNK, MCSR, install_fp32_norm_service, rmsnorm_weights


class Qwen2RMSNorm(torch.nn.Module):
    """Tiny fixture with the Qwen2 RMSNorm arithmetic and a BF16 parameter."""

    def __init__(self, width, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width, dtype=torch.bfloat16, device=device))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        v = x.float()
        v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * v.to(x.dtype)


class TinyBody(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embedding = torch.nn.Embedding(37, 16, device=device, dtype=torch.bfloat16)
        self.projection = torch.nn.Linear(16, 16, bias=False, device=device, dtype=torch.bfloat16)
        self.norm = Qwen2RMSNorm(16, device)
        self.gain = torch.nn.Parameter(torch.zeros(3, device=device, dtype=torch.bfloat16))

    def forward(self, input_ids, use_cache=False):
        h = self.embedding(input_ids)
        # Causal dependence so a different earlier token changes hidden states.
        h = self.projection(h.float().cumsum(dim=1).to(h.dtype))
        # Includes an entirely zero-gradient parameter to test that path.
        return SimpleNamespace(last_hidden_state=self.norm(h) + (self.gain * 0).sum())


class TinyLM(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = TinyBody(device)
        self.head = torch.nn.Linear(16, 37, bias=False, device=device, dtype=torch.bfloat16)
        self.head.weight = self.model.embedding.weight

    def get_output_embeddings(self):
        return self.head

    def get_input_embeddings(self):
        return self.model.embedding

    def gradient_checkpointing_enable(self):
        pass


MIXED = [
    {"prompt_ids": [1, 2, 3], "comp_ids": [[4, 5, 8], [6, 2], [7]],
     "advs": [2 / 3, -1 / 3, -1 / 3]},
    {"prompt_ids": [2, 5], "comp_ids": [[9, 2, 1], [1, 3]], "advs": [-0.5, 0.5]},
]


def verify_arms(device):
    reports = []
    for arm in MCSR_ARMS:
        for mode in (("bf16", "fp32") if arm in NORM_ARMS else ("bf16",)):
            torch.manual_seed(724)
            model = TinyLM(device)
            ps = list(model.parameters())
            state = create_arm_state(arm, ps, model, mode)
            install_fp32_norm_service(model, state)
            p = sum(w.numel() for w in ps)
            norms = {w.data_ptr() for w in rmsnorm_weights(model)} if arm in NORM_ARMS else set()
            expected = 2 * p
            for w in ps:
                if w.data_ptr() in norms or arm == "M_coupled":
                    expected += 4 * w.numel()
                elif arm in ("MCSR_int4", "MCSR_int4_norm_fp32"):
                    expected += (w.numel() + 1) // 2
                elif arm == "MCSR_int8":
                    expected += w.numel()
            assert state.state_bytes == expected
            on_device = 2 * p if device.type == "cuda" and arm == "M_coupled" else expected
            assert state.state_bytes_on(device) == on_device
            initial = state.checkpoint()
            zero_records = [dict(r, advs=[0.0] * len(r["advs"])) for r in MIXED]
            _, frac, did, stats = apply_step(arm, model, zero_records, None, 0.05, 61, 0,
                                             rows=11, state=state)
            assert not did and frac == 0 and stats["bit_changed_count"] == 0
            for before, after in zip(initial, state.checkpoint()):
                for key in before:
                    if before[key] is not None:
                        assert torch.equal(before[key], after[key])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            norm, frac, did, metrics = apply_step(arm, model, MIXED, None, 0.05, 61, 1,
                                                 rows=11, state=state)
            assert did and norm > 0 and frac > 0
            assert metrics["state_bytes"] == expected
            assert metrics["reverse_flip_count"] == metrics["saturated_count"] == 0
            assert metrics["master_transfer_bytes"] == (
                12 * p if device.type == "cuda" and arm == "M_coupled" else 0)
            assert all(t.grad is None and torch.isfinite(t).all() for t in ps)
            assert torch.equal(model.model.gain, torch.zeros_like(model.model.gain))
            assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
            reports.append({"arm": arm, "norm_service": mode, "parameters": p,
                            **metrics, "step_changed_frac": frac,
                            "peak_alloc_GiB": torch.cuda.max_memory_allocated(device) / 2**30
                            if device.type == "cuda" else None})
    return reports


def verify_cuda_replay(device):
    """Compare two CUDA streams/tilings with a CPU reference, including subnormals."""
    pattern = torch.tensor([0x0000, 0x8000, 0x0001, 0x8001, 0x3F7F, 0x3F80, 0xBF80,
                            0xBF7F, 0x7F7E, 0xFF7F], dtype=torch.int32).short().view(torch.bfloat16)
    w0 = pattern.repeat(52)[:-1].clone()
    d0 = torch.tensor([2**-140, -2**-140, -2**-133, 2**-133, 0.005, -0.007, -0.003,
                       0.008, 2**121, 0]).repeat(52)[:-1].clone()
    d1 = -d0
    stream = torch.cuda.Stream(device=device)
    for L in (8, 128, float("inf")):
        cpu = MCSR([w0.clone()], L, chunk_size=239)
        gpu = MCSR([w0.to(device)], L, chunk_size=257)
        d_gpu = (d0.to(device), d1.to(device))
        stream.wait_stream(torch.cuda.current_stream(device))
        for step, d in enumerate((d0, d1), 1):
            expected = cpu.update_([d], seed=547, step=step)
            with torch.cuda.stream(stream):
                actual = gpu.update_([d_gpu[step - 1]], seed=547, step=step)
            assert actual == expected, (L, step, actual, expected)
        for a, b in zip(cpu.checkpoint(), gpu.checkpoint()):
            for key, v in a.items():
                if v is not None:
                    dtype = torch.int16 if v.dtype == torch.bfloat16 else torch.int32 if v.dtype == torch.float32 else v.dtype
                    assert torch.equal(v.view(dtype), b[key].view(dtype)), (L, key)


def verify_cuda_workspace(device):
    n = DEFAULT_CHUNK + 1
    reports = []
    for L in (8, 128, float("inf")):
        w = torch.ones(n, device=device, dtype=torch.bfloat16)
        d = torch.empty(n, device=device, dtype=torch.bfloat16)
        d[::2], d[1::2] = 2**-10, -2**-10
        state = MCSR([w], L)
        state.update_([d], seed=3, step=1)  # Warm the sampler before measuring.
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        live = torch.cuda.memory_allocated(device)
        start = time.perf_counter()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            stats = state.update_([d], seed=3, step=2)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device)
        assert stats["reverse_flip_count"] == stats["saturated_count"] == 0
        assert peak - live < 64 * 2**20, (L, peak - live)
        reports.append({"L": str(L), "elements": n, "workspace_peak_bytes": peak - live,
                        "seconds": elapsed, **stats})
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    if device.type == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is required for the live GPU smoke")
    locked_countdown_splits()
    start = time.perf_counter()
    report = {"torch": torch.__version__, "device": str(device),
              "cuda": torch.version.cuda,
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "arms": verify_arms(device)}
    if device.type == "cuda":
        verify_cuda_replay(device)
        report["stream_chunk_cpu_parity"] = True
        report["workspace"] = verify_cuda_workspace(device)
    report["elapsed_sec"] = time.perf_counter() - start
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
