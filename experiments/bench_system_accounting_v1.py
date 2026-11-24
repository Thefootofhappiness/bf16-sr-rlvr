#!/usr/bin/env python3
"""Clean Standalone Per-Process Peak GPU VRAM + End-to-End Delta-Sync Benchmark.

Fixes every audit item from astra Req #4:
- Each memory mode runs in a completely fresh Python subprocess (zero CUDA allocator carryover).
- Exact BF16 stochastic rounding writeback (`sr_cast_tensor_gpu`) in all SR modes.
- Tied `embed_tokens` / `lm_head` preserved identically across ALL modes.
- Exact global L2 gradient norm clipping (`max_norm=1.0`) in `gpu_sgd_master_8p` and
  `gpu_sgd_masterless_sr_4p` (100% mathematically equivalent to closed-loop `S_full_sr`).
- Streaming hooked mode (`gpu_sgd_masterless_hooked_sr`) explicitly labeled with per-param clip
  and tied-weight-safe hook registration.
- Real end-to-end delta-sync pack/compress/decompress/apply timing at empirically measured churn
  rates (S_full_sr 0.442%, M_faircast 4.420%, plus full BF16 broadcast 100%).
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_unique_params(model):
    seen = {}
    out = []
    for name, p in model.named_parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen[ptr] = name
            out.append((name, p))
    return out


@torch.no_grad()
def sr_cast_tensor_gpu(x_fp32: torch.Tensor, rng_gen: torch.Generator, inf_pos: torch.Tensor, inf_neg: torch.Tensor) -> torch.Tensor:
    orig_shape = x_fp32.shape
    xf = x_fp32.reshape(-1)
    n_tot = xf.numel()
    chunk = 32_000_000
    out = torch.empty(n_tot, dtype=torch.bfloat16, device=x_fp32.device)
    for s in range(0, n_tot, chunk):
        e = min(s + chunk, n_tot)
        y = xf[s:e]
        r0 = y.bfloat16()
        r0_f = r0.float()
        le = r0_f <= y
        r_up = torch.nextafter(r0, inf_pos)
        r_dn = torch.nextafter(r0, inf_neg)
        lo = torch.where(le, r0, r_dn)
        hi = torch.where(le, r_up, r0)
        lo_f = lo.float()
        hi_f = hi.float()
        span = hi_f - lo_f
        prob_hi = torch.where(span > 0, (y - lo_f) / torch.clamp(span, min=1e-45), torch.zeros_like(y))
        u = torch.rand(y.shape, generator=rng_gen, device=y.device, dtype=torch.float32)
        out[s:e] = torch.where(u < prob_hi, hi, lo)
    return out.view(orig_shape)


def run_worker(mode: str, model_path: str, seq_len: int, batch_size: int, n_steps: int, eta: float):
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    rng_gen = torch.Generator(device=device).manual_seed(1234)
    inf_pos = torch.tensor(float("inf"), dtype=torch.bfloat16, device=device)
    inf_neg = torch.tensor(float("-inf"), dtype=torch.bfloat16, device=device)

    result = {
        "mode": mode,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "oom": False,
        "error": None,
    }

    try:
        load_dtype = torch.float32 if mode == "gpu_adamw_fp32_16p" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=load_dtype, attn_implementation="sdpa"
        ).to(device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.train()

        uniq_params = get_unique_params(model)
        n_params = sum(p.numel() for _, p in uniq_params)
        tied_lm_head = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        result["n_params"] = n_params
        result["tied_lm_head"] = tied_lm_head

        opt = None
        fp32_master = None
        hooks = []

        if mode == "gpu_adamw_fp32_16p":
            opt = torch.optim.AdamW([p for _, p in uniq_params], lr=1e-6)
        elif mode == "gpu_adamw_bf16_master_16p":
            fp32_master = [p.detach().float().clone().requires_grad_(True) for _, p in uniq_params]
            opt = torch.optim.AdamW(fp32_master, lr=1e-6)
        elif mode == "gpu_sgd_master_8p":
            fp32_master = [p.detach().float().clone() for _, p in uniq_params]
        elif mode == "gpu_sgd_masterless_sr_4p":
            pass
        elif mode == "gpu_sgd_masterless_hooked_sr_2p":
            # Streaming post-accumulate-grad hook with per-parameter clip (explicitly documented);
            # registers once per unique parameter (safe for tied embed_tokens/lm_head).
            def make_hook():
                def _hook(p: torch.Tensor):
                    if p.grad is None:
                        return
                    with torch.no_grad():
                        g = p.grad.float()
                        gn = torch.linalg.vector_norm(g)
                        scale = min(1.0, 1.0 / (gn.item() + 1e-12))
                        target = p.data.float() - (eta * scale) * g
                        p.data.copy_(sr_cast_tensor_gpu(target, rng_gen, inf_pos, inf_neg))
                        p.grad = None
                return _hook

            for _, p in uniq_params:
                hooks.append(p.register_post_accumulate_grad_hook(make_hook()))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        torch.cuda.synchronize()
        static_alloc = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        input_ids = torch.randint(10, 10000, (batch_size, seq_len), device=device)
        step_times = []

        for step in range(n_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Compute logits and FP32 LM-head cross-entropy (matching closed-loop script)
            out = model(input_ids=input_ids)
            logits = out.logits[:, :-1, :].float()
            targets = input_ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()

            if mode == "gpu_adamw_fp32_16p":
                torch.nn.utils.clip_grad_norm_([p for _, p in uniq_params], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            elif mode == "gpu_adamw_bf16_master_16p":
                for (_, p), m in zip(uniq_params, fp32_master):
                    m.grad = p.grad.float()
                    p.grad = None
                torch.nn.utils.clip_grad_norm_(fp32_master, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for (_, p), m in zip(uniq_params, fp32_master):
                        p.data.copy_(sr_cast_tensor_gpu(m.data, rng_gen, inf_pos, inf_neg))
            elif mode == "gpu_sgd_master_8p":
                total_sq = torch.zeros((), dtype=torch.float64, device=device)
                for _, p in uniq_params:
                    if p.grad is not None:
                        total_sq += p.grad.float().pow(2).sum(dtype=torch.float64)
                gn = math.sqrt(float(total_sq.item()))
                scale = min(1.0, 1.0 / (gn + 1e-12))
                with torch.no_grad():
                    for (_, p), m in zip(uniq_params, fp32_master):
                        if p.grad is not None:
                            m.sub_(p.grad.float(), alpha=eta * scale)
                            p.data.copy_(sr_cast_tensor_gpu(m, rng_gen, inf_pos, inf_neg))
                            p.grad = None
            elif mode == "gpu_sgd_masterless_sr_4p":
                total_sq = torch.zeros((), dtype=torch.float64, device=device)
                for _, p in uniq_params:
                    if p.grad is not None:
                        total_sq += p.grad.float().pow(2).sum(dtype=torch.float64)
                gn = math.sqrt(float(total_sq.item()))
                scale = min(1.0, 1.0 / (gn + 1e-12))
                with torch.no_grad():
                    for _, p in uniq_params:
                        if p.grad is not None:
                            target = p.data.float() - (eta * scale) * p.grad.float()
                            p.data.copy_(sr_cast_tensor_gpu(target, rng_gen, inf_pos, inf_neg))
                            p.grad = None
            elif mode == "gpu_sgd_masterless_hooked_sr_2p":
                pass  # already updated & freed inside post_accumulate_grad_hook

            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

        for h in hooks:
            h.remove()

        result["static_alloc_GiB"] = round(static_alloc / (1024**3), 4)
        result["peak_alloc_GiB"] = round(torch.cuda.max_memory_allocated(device) / (1024**3), 4)
        result["peak_reserved_GiB"] = round(torch.cuda.max_memory_reserved(device) / (1024**3), 4)
        result["step_times_sec"] = [round(x, 4) for x in step_times]
        result["mean_step_sec_after_warmup"] = round(float(np.mean(step_times[1:])), 4) if len(step_times) > 1 else round(step_times[0], 4)
        result["exact_global_clip"] = (mode != "gpu_sgd_masterless_hooked_sr_2p")
    except torch.cuda.OutOfMemoryError as e:
        result["oom"] = True
        result["peak_alloc_GiB"] = round(torch.cuda.max_memory_allocated(device) / (1024**3), 4)
        result["peak_reserved_GiB"] = round(torch.cuda.max_memory_reserved(device) / (1024**3), 4)
        result["error"] = str(e)[:240]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:240]}"

    print("WORKER_JSON:" + json.dumps(result))


def run_delta_sync_bench(model_path: str):
    """Real end-to-end GPU delta extraction + pack/zlib + unpack/scatter benchmark on actual Qwen2.5-Math-1.5B weights."""
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    uniq_params = get_unique_params(model)
    flat_base = torch.cat([p.detach().reshape(-1) for _, p in uniq_params])
    n_params = flat_base.numel()
    full_bytes = n_params * 2

    rng = np.random.default_rng(42)
    inf_pos = torch.tensor(float("inf"), dtype=torch.bfloat16, device=device)

    out = {
        "n_params": int(n_params),
        "full_bf16_MiB": round(full_bytes / (1024**2), 3),
        "scenarios": {},
    }

    # Measure full BF16 GPU->CPU->GPU transfer baseline
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cpu_full = flat_base.to("cpu", non_blocking=False)
    torch.cuda.synchronize()
    t_d2h_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = cpu_full.to(device, non_blocking=False)
    torch.cuda.synchronize()
    t_h2d_full = time.perf_counter() - t0
    out["full_bf16_transfer"] = {
        "d2h_ms": round(t_d2h_full * 1000, 2),
        "h2d_ms": round(t_h2d_full * 1000, 2),
        "net_10gbps_ms": round((full_bytes * 8 / 10e9) * 1000, 2),
        "net_25gbps_ms": round((full_bytes * 8 / 25e9) * 1000, 2),
    }
    del cpu_full

    # Compare empirically measured 3-seed mean step churn rates:
    # S_full_sr: 0.442% ; M_faircast: 4.420% ; plus vLLM seed=42 M_faircast: 6.185%
    for label, frac in [
        ("S_full_sr_3seed_mean_0.442pct", 0.00442),
        ("M_faircast_3seed_mean_4.420pct", 0.04420),
        ("M_faircast_vllm_s42_6.185pct", 0.06185),
    ]:
        # Construct realistic 1-ULP changed copy at exact fraction `frac`
        k = int(round(n_params * frac))
        idx_np = np.sort(rng.choice(n_params, size=k, replace=False)).astype(np.int64)
        idx_gpu = torch.from_numpy(idx_np).to(device)
        flat_new = flat_base.clone()
        flat_new[idx_gpu] = torch.nextafter(flat_new[idx_gpu], inf_pos)

        # 1) GPU diff + sparse COO extraction + D2H
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        diff_mask = flat_new.view(torch.int16) != flat_base.view(torch.int16)
        nz_idx = torch.nonzero(diff_mask, as_tuple=False).squeeze(1).to(torch.int32)
        nz_val = flat_new.view(torch.int16)[nz_idx.long()]
        nz_idx_cpu = nz_idx.cpu().numpy()
        nz_val_cpu = nz_val.cpu().numpy()
        torch.cuda.synchronize()
        t_extract_d2h = time.perf_counter() - t0

        raw_coo_bytes = nz_idx_cpu.nbytes + nz_val_cpu.nbytes

        # 2) Delta-gap encode sorted int32 indices + zlib level 1 (fast realtime sync) & level 6
        t0 = time.perf_counter()
        gaps = np.empty_like(nz_idx_cpu)
        gaps[0] = nz_idx_cpu[0]
        gaps[1:] = nz_idx_cpu[1:] - nz_idx_cpu[:-1]
        comp_idx_l1 = zlib.compress(gaps.tobytes(), level=1)
        comp_val_l1 = zlib.compress(nz_val_cpu.tobytes(), level=1)
        t_pack_l1 = time.perf_counter() - t0
        wire_bytes_l1 = len(comp_idx_l1) + len(comp_val_l1)

        t0 = time.perf_counter()
        gaps_dec = np.frombuffer(zlib.decompress(comp_idx_l1), dtype=np.int32)
        idx_dec = np.cumsum(gaps_dec, dtype=np.int64)
        val_dec = np.frombuffer(zlib.decompress(comp_val_l1), dtype=np.int16)
        t_unpack_l1 = time.perf_counter() - t0

        # 3) H2D + scatter-apply on receiver GPU & exact bit-match check
        flat_recv = flat_base.clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx_r = torch.from_numpy(idx_dec).to(device)
        val_r = torch.from_numpy(val_dec.copy()).to(device)
        flat_recv.view(torch.int16)[idx_r] = val_r
        torch.cuda.synchronize()
        t_h2d_apply = time.perf_counter() - t0
        bit_exact = bool(torch.equal(flat_recv.view(torch.int16), flat_new.view(torch.int16)))

        # Also check raw COO without compression (zero CPU compress latency, pure GPU->NIC->GPU)
        out["scenarios"][label] = {
            "churn_frac": frac,
            "n_changed": int(len(nz_idx_cpu)),
            "raw_coo_MiB": round(raw_coo_bytes / (1024**2), 3),
            "zlib_l1_MiB": round(wire_bytes_l1 / (1024**2), 3),
            "extract_d2h_ms": round(t_extract_d2h * 1000, 2),
            "pack_zlib_l1_ms": round(t_pack_l1 * 1000, 2),
            "unpack_zlib_l1_ms": round(t_unpack_l1 * 1000, 2),
            "h2d_apply_ms": round(t_h2d_apply * 1000, 2),
            "e2e_raw_coo_10gbps_ms": round((t_extract_d2h + t_h2d_apply + raw_coo_bytes * 8 / 10e9) * 1000, 2),
            "e2e_raw_coo_25gbps_ms": round((t_extract_d2h + t_h2d_apply + raw_coo_bytes * 8 / 25e9) * 1000, 2),
            "e2e_zlib_l1_10gbps_ms": round((t_extract_d2h + t_pack_l1 + t_unpack_l1 + t_h2d_apply + wire_bytes_l1 * 8 / 10e9) * 1000, 2),
            "bit_exact_verified": bit_exact,
        }
        del flat_new, flat_recv, idx_gpu, nz_idx, nz_val
        torch.cuda.empty_cache()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker-mode", default="")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.1)
    args = ap.parse_args()

    if args.worker_mode:
        run_worker(args.worker_mode, args.model, args.seq_len, args.batch_size, args.n_steps, args.eta)
        return

    modes = [
        "gpu_adamw_fp32_16p",
        "gpu_adamw_bf16_master_16p",
        "gpu_sgd_master_8p",
        "gpu_sgd_masterless_sr_4p",
        "gpu_sgd_masterless_hooked_sr_2p",
    ]

    summary = {
        "meta": {
            "model": args.model,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_GiB": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 3),
            "torch_version": torch.__version__,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "n_steps": args.n_steps,
            "eta": args.eta,
            "tf32": False,
        },
        "vram_modes": {},
    }

    for mode in modes:
        print(f"=== Running isolated subprocess: {mode} ===", flush=True)
        cmd = [
            sys.executable,
            __file__,
            "--model", args.model,
            "--out", args.out,
            "--worker-mode", mode,
            "--seq-len", str(args.seq_len),
            "--batch-size", str(args.batch_size),
            "--n-steps", str(args.n_steps),
            "--eta", str(args.eta),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        parsed = None
        for line in proc.stdout.splitlines():
            if line.startswith("WORKER_JSON:"):
                parsed = json.loads(line[len("WORKER_JSON:"):])
        if parsed is None:
            parsed = {"mode": mode, "oom": False, "error": f"No JSON output; rc={proc.returncode}; stderr={proc.stderr[-300:]}"}
        summary["vram_modes"][mode] = parsed
        print(json.dumps(parsed), flush=True)

    print("=== Running end-to-end Delta-Sync benchmark ===", flush=True)
    summary["delta_sync"] = run_delta_sync_bench(args.model)
    print(json.dumps(summary["delta_sync"], indent=2), flush=True)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"SAVED: {args.out}", flush=True)


if __name__ == "__main__":
    import math
    main()
