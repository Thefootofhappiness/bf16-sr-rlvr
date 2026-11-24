#!/usr/bin/env python3
"""Wave 2A (v2.1, sol-audited): Truly Matched (FP32 head + top_k=0) 1.5B GRPO + OOD (GSM8K) + WikiText-2 PPL + Clean Frozen Master Audit.

v2.2 adds the packed int4 and int8 MCSR update arms.

Fixes applied from gpt-6.2-sol audit (2026-11-20):
- P0: Removed @torch.no_grad() from apply_matched_step; forward + hidden slice run inside torch.enable_grad() with assert h.requires_grad.
- P0: Core module path searches both SCRIPT_DIR/bf16_sr_core and SCRIPT_DIR/../bf16_sr_core; logs exact __file__ + SHA256.
- P1: Balanced-brace last-\boxed{} GSM8K grader supporting \frac, unicode minus, scientific notation, unit suffixes, exact Fraction compare.
- P1: Exact update parity across all arms: d = gb.float().mul_(alpha); target.add_(d) (no FMA/non-FMA divergence).
- P1: Atomic save of main Step-200 JSON + metadata BEFORE running frozen Step-200 master audit (zero loss on audit OOM).
- P1: Clean FP32-backbone reference anchor (Step-0 BF16 weights upcast to FP32) for frozen master audit so exact FP32 master vs upcast RTNE vs upcast 5x SR are compared on identical FP32 backbone (plus BF16 deployment drift separately).
- P2: Enforced 0 < stride < max_len and assert tok_cnt == n - 1 in WikiText-2 PPL; CUDA-synchronized stage timings.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(SCRIPT_DIR, "bf16_sr_core"), os.path.join(os.path.dirname(SCRIPT_DIR), "bf16_sr_core")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

import bf16_sr
import mcsr_step
import sampling_utils
import sr_step
from bf16_sr import sr_into
from mcsr_step import MCSR
from sampling_utils import generate_fp32_head
from sr_step import StepGrads, blocks, grad_sqnorm, head_tiles

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

PROMPT_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem: "
PROMPT_SUFFIX = "\nSolution:"
MCSR_ARMS = {"MCSR_int4": 8, "MCSR_int8": 128, "M_coupled": mcsr_step.MASTER}
MCSR_COUNTS = ("reverse_flip_count", "saturated_count", "swallowed_count")
ADAM_ARMS = ("AdamW_fp32", "AdamW_faircast_sr", "AdamW_coupled_sr")
MASTER_ARMS = ("M_faircast", "M_rtne", "M_rtne_cl", "M_coupled", *ADAM_ARMS)
EXPLICIT_OPENER = r"(?:####|final answer\s*(?:is|:)|the answer is)"
EXPLICIT_MARKER = EXPLICIT_OPENER + r"\s*([^\n\.]+)"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


_PROMPT_STYLE = "raw"
_PROMPT_TOKENIZER = None


def format_prompt(problem: str) -> str:
    if _PROMPT_STYLE == "chat_template" and _PROMPT_TOKENIZER is not None:
        return _PROMPT_TOKENIZER.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful mathematical reasoning assistant."},
                {"role": "user", "content": PROMPT_PREFIX + problem},
            ],
            tokenize=False,
            add_generation_prompt=True,
            date_string="26 Jul 2024",
        )
    return PROMPT_PREFIX + problem + PROMPT_SUFFIX


def extract_last_boxed(text: str):
    """Inspect the last \\boxed{ opener, even when it is unclosed."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    value, closed = _boxed_at(text, idx)
    return value if closed else None


def _boxed_at(text, idx):
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip(), depth == 0


def _replace_latex_frac(s: str) -> str:
    """Replace balanced \\frac{a}{b} or \\dfrac{a}{b} with ((a)/(b))."""
    for cmd in ("\\dfrac", "\\tfrac", "\\frac"):
        while True:
            idx = s.find(cmd)
            if idx < 0:
                break
            p = idx + len(cmd)
            while p < len(s) and s[p].isspace():
                p += 1
            args = []
            ok = True
            for _ in range(2):
                if p >= len(s) or s[p] != "{":
                    ok = False
                    break
                p += 1
                depth = 1
                start = p
                while p < len(s) and depth > 0:
                    if s[p] == "{":
                        depth += 1
                    elif s[p] == "}":
                        depth -= 1
                    p += 1
                if depth != 0:
                    ok = False
                    break
                args.append(s[start : p - 1].strip())
                while p < len(s) and s[p].isspace():
                    p += 1
            if not ok:
                break
            s = s[:idx] + f"(({args[0]})/({args[1]}))" + s[p:]
    return s


def parse_gsm8k_number(raw: str):
    """Parse exact numeric value (Fraction) from a GSM8K boxed candidate or explicit final answer string."""
    if raw is None:
        return None
    s = raw.strip()
    if "=" in s:
        s = s.split("=")[-1].strip()
    s = s.replace("−", "-").replace("–", "-").replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{[^{}]*\}(?:\s*\^\s*(?:\{[^{}]*\}|\d+))?", "", s)
    s = re.sub(r"\\mathrm\{[^{}]*\}(?:\s*\^\s*(?:\{[^{}]*\}|\d+))?", "", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = _replace_latex_frac(s)
    # Remove thousands commas inside numbers (e.g., 1,234 -> 1234)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = s.strip()
    # Allow optional trailing unit text (words/spaces/carets like "cm^2", "dollars") ONLY after a leading number/fraction
    # Match exact fraction ((a)/(b)) or a/b
    m_frac = re.match(r"^\s*\(?\s*\(?\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*\)?\s*/\s*\(?\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*\)?\s*\)?(?:\s+[A-Za-z].*)?$", s)
    if m_frac:
        try:
            den = Fraction(m_frac.group(2))
            if den != 0:
                return Fraction(m_frac.group(1)) / den
        except Exception:
            return None
    # Match integer, decimal (.5, 0.5), or scientific notation (1e3, 1.5e-2), optional unit suffix
    m_num = re.match(r"^\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)(?:\s*(?:[A-Za-z][A-Za-z0-9\s\^\*/\-]*|))?$", s)
    if m_num:
        try:
            return Fraction(m_num.group(1))
        except Exception:
            return None
    return None


def extract_gsm8k_gold(ans_str: str) -> str:
    m = re.search(r"####\s*([-+]?[0-9,\.]+)", ans_str)
    return m.group(1).replace(",", "").strip() if m else ans_str.strip()


def grade_gsm8k_batch(preds, golds):
    out = []
    parse_fail = 0
    for p, g in zip(preds, golds):
        try:
            gv = Fraction(g.replace(",", "").strip())
            cand = extract_last_boxed(p)
            if cand is None:
                # Fallback ONLY on explicit final answer markers (#### or "Final answer:" / "The answer is")
                m = re.search(EXPLICIT_MARKER, p, re.IGNORECASE)
                cand = m.group(1).strip() if m else None
            pv = parse_gsm8k_number(cand)
            if pv is None:
                parse_fail += 1
                out.append(False)
            else:
                out.append(pv == gv)
        except Exception:
            parse_fail += 1
            out.append(False)
    return out


def gsm8k_grade_trace(text, gold):
    """Explain the legacy decision, including markers it did not select."""
    boxes = []
    for match in re.finditer(r"\\boxed\{", text):
        value, closed = _boxed_at(text, match.start())
        boxes.append({"offset": match.start(), "closed": closed, "value": value})
    pattern = re.compile(EXPLICIT_MARKER, re.IGNORECASE)
    markers = []
    for opener in re.finditer(EXPLICIT_OPENER, text, re.IGNORECASE):
        match = pattern.match(text, opener.start())
        markers.append({"offset": opener.start(), "marker": opener.group(),
                        "candidate": match.group(1).strip() if match else None})
    selected = pattern.search(text)
    if boxes and boxes[-1]["closed"]:
        branch, cand = "boxed", boxes[-1]["value"]
    elif selected:
        branch, cand = "explicit_marker", selected.group(1).strip()
    else:
        branch, cand = "none", None
    try:
        Fraction(gold.replace(",", "").strip())
        parsed = parse_gsm8k_number(cand)
    except Exception:
        parsed = None
    return {"all_boxed": boxes, "explicit_markers": markers, "selected_branch": branch,
            "parsed_value": None if parsed is None else str(parsed)}


def load_math_grader(pydeps_dir: str):
    if pydeps_dir and pydeps_dir not in sys.path:
        sys.path.insert(0, pydeps_dir)
    from math_grader import grade_math

    err_count = [0]

    def grade_batch(preds, golds, *, diagnostics=None):
        out = []
        for p, g in zip(preds, golds):
            try:
                result = grade_math(p, g)
                out.append(bool(result["correct_math_verify"]))
            except Exception:
                err_count[0] += 1
                out.append(False)
                result = None
            if diagnostics is not None:
                diagnostics.append(result)
        return out

    return grade_batch, err_count


def get_unique_params(model):
    ps = [p for p in model.parameters() if p.requires_grad]
    ptrs = [p.untyped_storage().data_ptr() for p in ps]
    if len(set(ptrs)) != len(ptrs):
        raise ValueError("shared storage across distinct Parameter objects is unsupported")
    return ps


def create_mcsr_state(arm, ps, chunk_size=mcsr_step.DEFAULT_CHUNK):
    master_device = "cpu" if arm == "M_coupled" else None
    return MCSR(ps, MCSR_ARMS[arm], master_device=master_device, chunk_size=chunk_size)


def create_adam_state(masters, wd):
    return {
        "m_cpu": [torch.zeros_like(m) for m in masters],
        "v_cpu": [torch.zeros_like(m) for m in masters],
        "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "wd": wd, "t": 0,
    }


def coupled_master_export(old_master, old_serving, new_master, seed, step, param_idx, offset):
    """Project an already computed FP32 target using Countdown's L=inf kernel."""
    limit = torch.finfo(torch.bfloat16).max
    x = old_master.clamp(-limit, limit).view(torch.int32)
    y = new_master.clamp(-limit, limit).view(torch.int32)
    word = old_serving.view(torch.int16)
    old_rank = mcsr_step._rank(word.int())
    near, away, _, old_here, keep, rem_near, rem_away = mcsr_step._coupling_masses(old_rank, x, y, 0)
    stay_key = mcsr_step._key(seed, step, param_idx, "stay")
    move_key = mcsr_step._key(seed, step, param_idx, "move")
    leave = mcsr_step._uniform(old_here, offset, stay_key, bounds=old_here) >= keep
    remaining = rem_near.add_(rem_away).clamp_min_(1)
    moving_away = mcsr_step._uniform(remaining, offset, move_key, bounds=remaining) < rem_away
    new_rank = torch.where(moving_away, away, near)
    return torch.where(leave, mcsr_step._code(new_rank, y < 0).short(), word).view(torch.bfloat16)


def count_reverse_flips(old_master, new_master, old_serving, new_serving):
    return torch.count_nonzero(
        ((new_serving < old_serving) & (new_master > old_master))
        | ((new_serving > old_serving) & (new_master < old_master))
    )


@torch.inference_mode()
def matched_generate_batch(model, tokenizer, prompts, device, max_new_tokens: int, n_samples: int, temperature: float, seed: int, batch_size: int = 32):
    eos_id = tokenizer.eos_token_id
    exp_prompts = [p for p in prompts for _ in range(n_samples)]
    prompt_ids_uniq = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    all_comp_ids, all_texts, all_trunc = [], [], []

    for s in range(0, len(exp_prompts), batch_size):
        sub = exp_prompts[s : s + batch_size]
        enc = tokenizer(sub, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        in_len = input_ids.shape[1]

        out_ids = generate_fp32_head(
            model, input_ids, attn_mask, tokenizer,
            limit=max_new_tokens, temperature=temperature, seed=seed + s
        )
        gen_part = out_ids[:, in_len:].cpu().tolist()
        for seq in gen_part:
            if eos_id in seq:
                idx = seq.index(eos_id)
                comp = seq[: idx + 1]
                trunc = False
            else:
                comp = seq
                trunc = len(comp) >= max_new_tokens
            all_comp_ids.append(comp)
            all_texts.append(tokenizer.decode(comp, skip_special_tokens=True))
            all_trunc.append(trunc)

    return [
        {
            "prompt_ids": prompt_ids_uniq[i],
            "comp_ids": all_comp_ids[i * n_samples : (i + 1) * n_samples],
            "texts": all_texts[i * n_samples : (i + 1) * n_samples],
            "truncs": all_trunc[i * n_samples : (i + 1) * n_samples],
        }
        for i in range(len(prompts))
    ]


def apply_matched_step(arm: str, model, records, fp32_master_cpu, eta: float, seed: int, step: int, rows: int = 2048, chunk: int = 1 << 20, *, state=None, adam_state=None, master_telemetry=False):
    """NOTE: Do NOT decorate with @torch.no_grad(); forward + hidden slice MUST run with autograd enabled."""
    if arm in MCSR_ARMS and (state is None or state.L != MCSR_ARMS[arm]):
        raise ValueError(f"{arm} requires its matching MCSR state")
    if arm in ADAM_ARMS and (fp32_master_cpu is None or adam_state is None):
        raise ValueError(f"{arm} requires fp32_master_cpu and adam_state")
    model.train()
    model.gradient_checkpointing_enable()
    w_head = model.get_output_embeddings().weight
    tied = w_head is model.get_input_embeddings().weight
    nt = max(sum(len(c) for rec in records for c in rec["comp_ids"]), 1)

    if w_head.is_cuda:
        torch.cuda.synchronize()
    t_bwd0 = time.perf_counter()
    with StepGrads(model, host=False) as sink:
        for rec in records:
            prompt = rec["prompt_ids"]
            for c, adv in zip(rec["comp_ids"], rec["advs"]):
                if not c or abs(adv) < 1e-12:
                    continue
                ids = torch.tensor(prompt + c, device=w_head.device)
                with torch.enable_grad():
                    hidden = model.model(input_ids=ids[None], use_cache=False).last_hidden_state
                    h = hidden[0, len(prompt) - 1 : -1]
                assert h.requires_grad and h.shape[0] == len(c), "h must require grad for backward"
                head_tiles(h, w_head, ids[len(prompt) :], adv / nt, sink, tied, rows)
        if w_head.is_cuda:
            torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t_bwd0

        t_upd0 = time.perf_counter()
        with torch.no_grad():
            ps = sink.ps
            gs = [p.grad for p in ps]
            squares = [grad_sqnorm(g, chunk) for g in gs if g is not None]
            if not squares:
                return 0.0, 0.0, False, t_bwd, 0.0, {}
            norm = torch.stack(squares).sum().sqrt().item()
            if not math.isfinite(norm):
                raise FloatingPointError(f"nonfinite global norm at step {step}: {norm}")
            if norm == 0.0:
                for p in ps:
                    p.grad = None
                return 0.0, 0.0, False, t_bwd, 0.0, {}

            alpha = -eta * min(1.0, 1.0 / norm)
            if arm in MCSR_ARMS:
                weights = state.weights
                if len(ps) != len(weights) or any(p is not w for p, w in zip(ps, weights)):
                    raise RuntimeError("MCSR parameter order differs from the gradient collector")
                stats = state.update_(gs, seed=seed, step=step, alpha=alpha)
                for p in ps:
                    p.grad = None
                t_upd = time.perf_counter() - t_upd0
                counts = {key: stats[key] for key in MCSR_COUNTS}
                if arm == "M_coupled":
                    counts["coo_bytes_raw"] = 6 * stats["bit_changed_count"]
                return norm, stats["step_changed_frac"], True, t_bwd, t_upd, counts
            dev = ps[0].device
            track_master = master_telemetry and arm in MASTER_ARMS
            totals = torch.zeros(2 if track_master else 1, device=dev, dtype=torch.int64)
            changed_tot = totals[0]
            n_params = sum(p.numel() for p in ps)

            for j, (p, g) in enumerate(zip(ps, gs)):
                if g is None:
                    continue
                gen = torch.Generator(device=dev).manual_seed(seed * 1000003 + step * 1009 + j)
                w = p.view(-1)
                if arm == "S_full_sr":
                    for i, gb in blocks(g, chunk):
                        old = w[i : i + gb.numel()]
                        d = gb.float().mul_(alpha)
                        y = old.float().add_(d)
                        y.masked_fill_((d == 0) & (old.view(torch.int16) == -32768), -0.0)
                        del d
                        dst = torch.empty_like(old)
                        sr_into(y, dst, gen, chunk)
                        changed_tot.add_(torch.count_nonzero(dst.view(torch.int16) != old.view(torch.int16)))
                        old.copy_(dst)
                elif arm in ("M_faircast", "M_rtne", "M_rtne_cl"):
                    m_cpu = fp32_master_cpu[j].view(-1)
                    for i, gb in blocks(g, chunk):
                        old = w[i : i + gb.numel()]
                        d = gb.float().mul_(alpha)
                        m_blk = m_cpu[i : i + gb.numel()].to(dev)
                        if track_master:
                            old_master = m_blk.clone()
                        m_blk.add_(d)
                        del d
                        m_cpu[i : i + gb.numel()].copy_(m_blk.cpu())
                        if arm != "M_faircast":
                            dst = m_blk.bfloat16()
                        else:
                            dst = torch.empty_like(old)
                            sr_into(m_blk, dst, gen, chunk)
                        changed_tot.add_(torch.count_nonzero(dst.view(torch.int16) != old.view(torch.int16)))
                        if track_master:
                            totals[1].add_(count_reverse_flips(old_master, m_blk, old, dst))
                        old.copy_(dst)
                elif arm == "R_pure_bf16":
                    for i, gb in blocks(g, chunk):
                        old = w[i : i + gb.numel()]
                        d = gb.float().mul_(alpha)
                        dst = old.float().add_(d).bfloat16()
                        del d
                        changed_tot.add_(torch.count_nonzero(dst.view(torch.int16) != old.view(torch.int16)))
                        old.copy_(dst)
                elif arm in ADAM_ARMS:
                    clip_scale = min(1.0, 1.0 / norm)
                    beta1 = adam_state["beta1"]
                    beta2 = adam_state["beta2"]
                    eps = adam_state["eps"]
                    wd = adam_state["wd"] if p.ndim >= 2 else 0.0
                    t_adam = adam_state["t"] + 1
                    bc1 = 1.0 - (beta1 ** t_adam)
                    bc2_sqrt = math.sqrt(1.0 - (beta2 ** t_adam))
                    step_size = eta / bc1
                    m_cpu = fp32_master_cpu[j].view(-1)
                    exp_avg_cpu = adam_state["m_cpu"][j].view(-1)
                    exp_avg_sq_cpu = adam_state["v_cpu"][j].view(-1)
                    for i, gb in blocks(g, chunk):
                        sl = slice(i, i + gb.numel())
                        old = w[sl]
                        gf = gb.float().mul_(clip_scale)
                        m_b = exp_avg_cpu[sl].to(dev).mul_(beta1).add_(gf, alpha=1.0 - beta1)
                        v_b = exp_avg_sq_cpu[sl].to(dev).mul_(beta2).addcmul_(gf, gf, value=1.0 - beta2)
                        exp_avg_cpu[sl].copy_(m_b.cpu())
                        exp_avg_sq_cpu[sl].copy_(v_b.cpu())
                        w_b = m_cpu[sl].to(dev)
                        if track_master or arm == "AdamW_coupled_sr":
                            old_master = w_b.clone()
                        if wd != 0.0:
                            w_b.mul_(1.0 - eta * wd)
                        denom = v_b.sqrt().div_(bc2_sqrt).add_(eps)
                        w_b.addcdiv_(m_b, denom, value=-step_size)
                        m_cpu[sl].copy_(w_b.cpu())
                        if arm == "AdamW_fp32":
                            dst = w_b.bfloat16()
                        elif arm == "AdamW_faircast_sr":
                            dst = torch.empty_like(old)
                            sr_into(w_b, dst, gen, chunk)
                        else:
                            dst = coupled_master_export(old_master, old, w_b, seed, step, j, i)
                        changed_tot.add_(torch.count_nonzero(dst.view(torch.int16) != old.view(torch.int16)))
                        if track_master:
                            totals[1].add_(count_reverse_flips(old_master, w_b, old, dst))
                        old.copy_(dst)
                else:
                    raise ValueError(arm)
                p.grad = None
            if arm in ADAM_ARMS:
                adam_state["t"] += 1
            counts = {}
            if track_master:
                changed, reverse = totals.tolist()
                counts = {"coo_bytes_raw": 6 * changed, "reverse_flip_count": reverse}
            else:
                changed = changed_tot.item()
            changed_val = changed / max(n_params, 1)
            if w_head.is_cuda:
                torch.cuda.synchronize()
            t_upd = time.perf_counter() - t_upd0
            return norm, changed_val, True, t_bwd, t_upd, counts


@torch.inference_mode()
def cache_anchor_kl_prefix(model, kl_prefix_recs, device):
    model.eval()
    ref_h_cpu = []
    for rec in kl_prefix_recs:
        p_ids = rec["prompt_ids"]
        Lp = len(p_ids)
        for c_ids in rec["comp_ids"]:
            if not c_ids:
                continue
            full = torch.tensor([p_ids + c_ids], dtype=torch.long, device=device)
            h = model.model(input_ids=full, use_cache=False).last_hidden_state[0, Lp - 1 : -1].detach().cpu()
            ref_h_cpu.append(h)
    ref_hw_cpu = model.get_output_embeddings().weight.detach().cpu().clone()
    return ref_h_cpu, ref_hw_cpu


@torch.inference_mode()
def compute_kl_from_ref(model, ref_h_cpu, ref_hw_cpu, kl_prefix_recs, device):
    model.eval()
    ref_hw = ref_hw_cpu.to(device).float()
    cur_hw = model.get_output_embeddings().weight.detach().float()
    total_kl = 0.0
    total_tok = 0
    idx = 0
    for rec in kl_prefix_recs:
        p_ids = rec["prompt_ids"]
        Lp = len(p_ids)
        for c_ids in rec["comp_ids"]:
            if not c_ids:
                continue
            full = torch.tensor([p_ids + c_ids], dtype=torch.long, device=device)
            cur_h = model.model(input_ids=full, use_cache=False).last_hidden_state[0, Lp - 1 : -1].float()
            ref_h = ref_h_cpu[idx].to(device).float()
            idx += 1
            T = cur_h.shape[0]
            for s in range(0, T, 64):
                e = min(s + 64, T)
                lp_ref = F.log_softmax(F.linear(ref_h[s:e], ref_hw), dim=-1)
                lp_cur = F.log_softmax(F.linear(cur_h[s:e], cur_hw), dim=-1)
                total_kl += float((lp_ref.exp() * (lp_ref - lp_cur)).sum().item())
                total_tok += e - s
    return total_kl / max(total_tok, 1)


@torch.inference_mode()
def reference_token_logps(model, reference, device):
    """Teacher-force the same observed completion tokens in both policies."""
    model.eval()
    head = model.get_output_embeddings()
    weight = head.weight.detach().float()
    bias = None if head.bias is None else head.bias.detach().float()
    batches = []
    for rec in reference:
        prompt = rec["prompt_ids"]
        for tokens in rec["comp_ids"]:
            if not tokens:
                continue
            ids = torch.tensor([prompt + tokens], dtype=torch.long, device=device)
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, len(prompt) - 1:-1]
            target = ids[0, len(prompt):]
            for start in range(0, len(tokens), 64):
                stop = start + 64
                logits = F.linear(hidden[start:stop].float(), weight, bias)
                batches.append(F.log_softmax(logits, dim=-1).gather(1, target[start:stop, None]).cpu())
    return torch.cat(batches).view(-1) if batches else torch.empty(0)


@torch.no_grad()
def load_fp32_master(model, masters):
    params = get_unique_params(model)
    if len(params) != len(masters):
        raise ValueError("FP32 model and master parameter counts differ")
    for param, master in zip(params, masters):
        if param.dtype != torch.float32 or param.shape != master.shape:
            raise ValueError("FP32 model and master parameters must have matching shapes")
        param.copy_(master)


def fp32_eval_model(model_path, device):
    # Loading a shadow model must not change subsequent training RNG state.
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, attn_implementation="sdpa"
        ).to(device)


@torch.no_grad()
def sticking_fraction(params, masters, initial, chunk=mcsr_step.DEFAULT_CHUNK):
    stuck = 0
    n_params = 0
    for param, master, init in zip(params, masters, initial):
        w = param.view(-1)
        m = master.view(-1)
        for start, w0 in blocks(init.view(-1), chunk):
            end = start + w0.numel()
            unchanged = w[start:end].cpu().view(torch.int16) == w0.view(torch.int16)
            stuck += torch.count_nonzero(unchanged & (m[start:end] != w0.float())).item()
        n_params += w.numel()
    return stuck / n_params


@torch.inference_mode()
def compute_wikitext_ppl(model, token_ids_1d: torch.Tensor, device, max_len: int = 1024, stride: int = 512):
    assert 0 < stride < max_len and token_ids_1d.numel() >= 2
    model.eval()
    head_w = model.get_output_embeddings().weight.detach().float()
    n = token_ids_1d.numel()
    nll_sum = 0.0
    tok_cnt = 0
    prev_end = 0
    for begin in range(0, n - 1, stride):
        end = min(begin + max_len, n)
        inp = token_ids_1d[begin:end].unsqueeze(0).to(device)
        h = model.model(input_ids=inp, use_cache=False).last_hidden_state[0, :-1].float()
        targets = inp[0, 1:]
        start_pos = max(0, (prev_end - begin) - 1) if begin > 0 else 0
        h_sc = h[start_pos:]
        t_sc = targets[start_pos:]
        for s in range(0, h_sc.shape[0], 64):
            e = min(s + 64, h_sc.shape[0])
            logits = F.linear(h_sc[s:e], head_w)
            nll_sum += float(F.cross_entropy(logits, t_sc[s:e], reduction="sum").item())
            tok_cnt += e - s
        prev_end = end
        if end == n:
            break
    assert tok_cnt == n - 1, f"PPL token count mismatch: {tok_cnt} vs {n - 1}"
    mean_nll = nll_sum / tok_cnt
    return {"split": "wikitext-2-raw-v1:validation", "mean_nll": mean_nll, "ppl": math.exp(min(mean_nll, 20.0)), "n_tokens": tok_cnt}


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(math.comb(n - c, k) / math.comb(n, k))


def save_eval_completions(path, items, batches, *, gsm8k):
    """Write actual graded completions in rollout order and return mode counts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".completions-", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            for mode, outs, flags, math_traces in batches:
                categories = Counter(dict.fromkeys(("no_candidate", "parse_fail", "wrong_value", "correct"), 0))
                branches = Counter()
                truncated = 0
                flat_idx = 0
                for item, out in zip(items, outs):
                    for k, (text, tokens, trunc) in enumerate(zip(out["texts"], out["comp_ids"], out["truncs"])):
                        ok = bool(flags[flat_idx])
                        trace = gsm8k_grade_trace(text, item["answer"])
                        if not gsm8k:
                            details = math_traces[flat_idx]
                            parsed = details["prediction_parse"] if details is not None else []
                            trace["parsed_value"] = parsed or None
                            trace["selected_branch"] = "math_verify"
                            has_candidate = bool(parsed or trace["all_boxed"] or trace["explicit_markers"]
                                                 or re.search(r"\d", text))
                            parse_failed = details is None or details["grade_status"] != "graded"
                        else:
                            has_candidate = trace["selected_branch"] != "none"
                            parse_failed = trace["parsed_value"] is None
                        if ok:
                            category = "correct"
                        elif not has_candidate:
                            category = "no_candidate"
                        elif parse_failed:
                            category = "parse_fail"
                        else:
                            category = "wrong_value"
                        record = {"item_idx": item["_idx"], "mode": mode, "sample_idx": k,
                                  "gold": item["answer"], "text": text, "n_tokens": len(tokens),
                                  "truncated": bool(trunc), **trace, "grade_ok": ok,
                                  "failure_category": category}
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        categories[category] += 1
                        branches[trace["selected_branch"]] += 1
                        truncated += int(trunc)
                        flat_idx += 1
                summary[mode] = {"n": flat_idx, "truncated": truncated,
                                 "failure_categories": dict(categories), "selected_branches": dict(branches)}
            stream.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    return summary


def evaluate_suite(model, tokenizer, items, grade_fn, device, max_tokens: int, n_samples: int, seed: int,
                   batch_size: int = 64, *, completions_path=None, grader_diag=None):
    prompts = [format_prompt(it["problem"]) for it in items]
    golds = [it["answer"] for it in items]
    is_gsm = grade_fn is grade_gsm8k_batch
    batches = []

    def grade(preds, answers, mode, outs):
        math_traces = [] if completions_path is not None and not is_gsm else None
        flags = (grade_fn(preds, answers) if math_traces is None
                 else grade_fn(preds, answers, diagnostics=math_traces))
        if completions_path is not None:
            batches.append((mode, outs, flags, math_traces))
        return flags

    t0 = time.perf_counter()
    outs_g = matched_generate_batch(model, tokenizer, prompts, device, max_tokens, n_samples=1, temperature=0.0, seed=seed, batch_size=batch_size)
    flags_g = grade([o["texts"][0] for o in outs_g], golds, "greedy", outs_g)

    per_item_p1, per_item_pk = [], []
    mean_len_s = 0.0
    trunc_rate_s = 0.0
    if n_samples > 0:
        outs_s = matched_generate_batch(model, tokenizer, prompts, device, max_tokens, n_samples=n_samples, temperature=0.6, seed=seed + 100000, batch_size=batch_size)
        preds_s = [txt for o in outs_s for txt in o["texts"]]
        golds_s = [g for g in golds for _ in range(n_samples)]
        flags_s = grade(preds_s, golds_s, "sample_k", outs_s)
        lens, truncs = [], []
        for i in range(len(items)):
            sub = flags_s[i * n_samples : (i + 1) * n_samples]
            c = sum(1 for x in sub if x)
            per_item_p1.append(c / n_samples)
            per_item_pk.append(pass_at_k(n_samples, c, n_samples))
            for comp, tr in zip(outs_s[i]["comp_ids"], outs_s[i]["truncs"]):
                lens.append(len(comp))
                truncs.append(1 if tr else 0)
        mean_len_s = float(np.mean(lens)) if lens else 0.0
        trunc_rate_s = float(np.mean(truncs)) if truncs else 0.0

    result = {
        "n_items": len(items),
        "n_samples": n_samples,
        "greedy_pass1": float(np.mean(flags_g)),
        "stoch_pass1": float(np.mean(per_item_p1)) if per_item_p1 else None,
        f"pass@{n_samples}": float(np.mean(per_item_pk)) if per_item_pk else None,
        "mean_len_stoch": mean_len_s,
        "trunc_rate_stoch": trunc_rate_s,
        "eval_sec": float(time.perf_counter() - t0),
        "per_item_greedy": [1 if x else 0 for x in flags_g],
        "per_item_p1": per_item_p1,
    }
    if completions_path is not None:
        summary = save_eval_completions(completions_path, items, batches, gsm8k=is_gsm)
        if grader_diag is not None:
            grader_diag.update(summary)
    return result


def save_final_checkpoint(path, model, params, state, masters, config, kl_sha, step):
    """Save physical BF16 weights and native packed residuals with parameter names."""
    if state is None:
        entries = [{"weight": p.detach().to("cpu", copy=True), "residual": None,
                    "master": None if masters is None else masters[i]}
                   for i, p in enumerate(params)]
    else:
        entries = state.checkpoint()
    indices = {id(p): i for i, p in enumerate(params)}
    names = {name: indices[id(p)] for name, p in model.named_parameters(remove_duplicate=False)
             if id(p) in indices}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "matched_ood_v2.2", "config": config, "step": step,
               "kl_sha": kl_sha, "name_to_index": names, "parameters": entries,
               "S_pack": [entry["residual"] for entry in entries] if state is not None
               and state.L != mcsr_step.MASTER else None}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".matched-", suffix=".pt", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(dict.fromkeys(
        ("anchor_only", "S_full_sr", "R_pure_bf16", *MASTER_ARMS, *MCSR_ARMS)
    )))
    ap.add_argument("--adamw-wd", type=float, default=0.01)
    ap.add_argument("--model", required=True)
    ap.add_argument("--math-data", required=True)
    ap.add_argument("--math-splits", required=True)
    ap.add_argument("--gsm8k-data", required=True)
    ap.add_argument("--wikitext-data", required=True)
    ap.add_argument("--pydeps", required=True)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gsm-max-tokens", type=int, default=512)
    ap.add_argument("--eval-n", type=int, default=8)
    ap.add_argument("--eval-bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mcsr-chunk-size", type=int, default=argparse.SUPPRESS,
                    help=f"MCSR update tile size (default: {mcsr_step.DEFAULT_CHUNK})")
    ap.add_argument("--prompt-style", choices=["raw", "chat_template"], default="raw")
    ap.add_argument("--eval-master-end", action="store_true", default=argparse.SUPPRESS,
                    help="Also evaluate the exact FP32 master in FP32 (greedy + 8 samples)")
    ap.add_argument("--save-eval-completions", type=Path, default=argparse.SUPPRESS,
                    help="Write held-out completions and original grading diagnostics as JSONL")
    ap.add_argument("--skip-math-eval", action="store_true", default=argparse.SUPPRESS,
                    help="Skip MATH evaluations, while still training and grading training rollouts normally")
    ap.add_argument("--save-final-ckpt", default=argparse.SUPPRESS,
                    help="Save final BF16 weights and packed MCSR state, if present")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    eval_master_end = getattr(args, "eval_master_end", False)
    completions_dir = getattr(args, "save_eval_completions", None)
    skip_math = getattr(args, "skip_math_eval", False)
    ckpt_path = getattr(args, "save_final_ckpt", None)
    master_arm = args.arm in MASTER_ARMS
    if eval_master_end and not master_arm:
        ap.error("--eval-master-end requires an FP32-master arm")
    config = vars(args).copy()
    if completions_dir is not None:
        config["save_eval_completions"] = str(completions_dir)

    core_modules = (bf16_sr, sr_step, sampling_utils)
    protocol = "matched_fp32_head_topk0_bit_level_sr_v2.1"
    if args.arm in MCSR_ARMS:
        args.mcsr_chunk_size = getattr(args, "mcsr_chunk_size", mcsr_step.DEFAULT_CHUNK)
        if args.mcsr_chunk_size < 1:
            ap.error("--mcsr-chunk-size must be positive")
        config["mcsr_chunk_size"] = args.mcsr_chunk_size
        config["mcsr_L"] = "inf" if args.arm == "M_coupled" else MCSR_ARMS[args.arm]
        core_modules += (mcsr_step,)
        protocol = "matched_fp32_head_topk0_bit_level_mcsr_v2.2"
    elif args.arm in ADAM_ARMS[1:]:
        if args.arm == "AdamW_coupled_sr":
            core_modules += (mcsr_step,)
        protocol = "matched_fp32_head_topk0_adamw_export_v2.2"
    if args.arm in ("M_coupled", *ADAM_ARMS[1:]):
        config["master_device"] = "cpu"
    core_manifest = {
        m.__name__: {"file": m.__file__, "sha256": sha256_file(m.__file__)}
        for m in core_modules
    }
    print("[core_manifest]", json.dumps(core_manifest), flush=True)

    t_start = time.perf_counter()
    device = torch.device("cuda:0")
    math_grade, math_err_cnt = load_math_grader(args.pydeps)

    with open(args.math_data) as f:
        m_rows = [json.loads(line) for line in f if line.strip()]
    with open(args.math_splits) as f:
        sp = json.load(f)
    train_items = [dict(m_rows[i], _idx=i) for i in sp["dev_ids"][:240]]
    dev_val_items = [dict(m_rows[i], _idx=i) for i in sp["dev_ids"][240:280]]
    kl_prefix_items = [dict(m_rows[i], _idx=i) for i in sp["dev_ids"][280:292]]
    math_test_items = [dict(m_rows[i], _idx=i) for i in sp["test_ids"]]

    with open(args.gsm8k_data) as f:
        g_rows = [json.loads(line) for line in f if line.strip()]
    rng_g = np.random.default_rng(42)
    g_idxs = sorted(rng_g.choice(len(g_rows), size=200, replace=False).tolist())
    gsm_test_items = [{"problem": g_rows[i]["question"], "answer": extract_gsm8k_gold(g_rows[i]["answer"]), "_idx": i} for i in g_idxs]

    def final_suites(eval_model, directory=None, *, n_samples=args.eval_n, batch_size=args.eval_bs, diag=None):
        math_ev = None if skip_math else evaluate_suite(
            eval_model, tokenizer, math_test_items, math_grade, device, args.max_tokens,
            n_samples, args.seed + 777, batch_size,
            completions_path=None if directory is None else directory / "math500_completions.jsonl",
        )
        gsm_ev = evaluate_suite(
            eval_model, tokenizer, gsm_test_items, grade_gsm8k_batch, device, args.gsm_max_tokens,
            n_samples, args.seed + 888, batch_size, grader_diag=diag,
            completions_path=None if directory is None else directory / "gsm8k_ood_completions.jsonl",
        )
        return {"math500_test": math_ev, "gsm8k_ood": gsm_ev}

    def scores_label(evals):
        labels = []
        for suite, name in (("math500_test", "MATH"), ("gsm8k_ood", "GSM8K")):
            ev = evals[suite]
            if ev is not None:
                stoch = ev["stoch_pass1"]
                labels.append(f"{name} stoch={stoch if stoch is None else format(stoch, '.4f')} "
                              f"(greedy={ev['greedy_pass1']:.4f})")
        return " | ".join(labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    global _PROMPT_STYLE, _PROMPT_TOKENIZER
    _PROMPT_STYLE = args.prompt_style
    _PROMPT_TOKENIZER = tokenizer

    with open(args.wikitext_data) as f:
        wt_text = f.read()
    wt_ids = torch.tensor(tokenizer.encode(wt_text, add_special_tokens=False), dtype=torch.long)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    ps = get_unique_params(model)
    n_params = sum(p.numel() for p in ps)
    anchor_bf16_cpu = [p.detach().to("cpu", copy=True) for p in ps] if master_arm else None
    fp32_master_cpu = ([w.float() for w in anchor_bf16_cpu]
                       if master_arm and args.arm != "M_coupled" else None)
    adam_state = create_adam_state(fp32_master_cpu, args.adamw_wd) if args.arm in ADAM_ARMS else None

    kl_outs = matched_generate_batch(
        model, tokenizer, [format_prompt(it["problem"]) for it in kl_prefix_items],
        device, max_new_tokens=512, n_samples=2, temperature=1.0, seed=9999, batch_size=24
    )
    kl_prefix_recs = [{"prompt_ids": o["prompt_ids"], "comp_ids": o["comp_ids"]} for o in kl_outs]
    kl_prefix_sha256 = hashlib.sha256(json.dumps(kl_prefix_recs).encode()).hexdigest()
    ref_h_cpu, ref_hw_bf16_cpu = cache_anchor_kl_prefix(model, kl_prefix_recs, device)

    wt0 = compute_wikitext_ppl(model, wt_ids, device)
    print(f"[start matched_v2.1 arm={args.arm} seed={args.seed}] kl_sha={kl_prefix_sha256[:12]} wt2_ppl={wt0['ppl']:.4f} (toks={wt0['n_tokens']})", flush=True)

    if args.arm == "anchor_only":
        gsm_diag = {}
        evals = final_suites(model, completions_dir, diag=gsm_diag)
        res = {
            "config": config,
            "core_manifest": core_manifest,
            "protocol": "matched_fp32_head_topk0_bit_level_sr_v2.1",
            "kl_prefix_sha256": kl_prefix_sha256,
            "n_params": n_params,
            "wikitext2": wt0,
            **evals,
            "math_grader_errors": math_err_cnt[0],
            "elapsed_sec": time.perf_counter() - t_start,
        }
        if completions_dir is not None:
            res["gsm8k_grader_diag"] = gsm_diag
        atomic_write_json(args.out, res)
        if ckpt_path:
            save_final_checkpoint(ckpt_path, model, ps, None, None, config, kl_prefix_sha256, 0)
        print(f"[DONE anchor] {scores_label(evals)} | WT2 PPL={wt0['ppl']:.4f}", flush=True)
        return

    state = None
    mcsr_idle = {}
    if args.arm in MCSR_ARMS:
        state = create_mcsr_state(args.arm, ps, args.mcsr_chunk_size)
        mcsr_idle = dict.fromkeys(MCSR_COUNTS, 0)
        mcsr_idle.update(state_bytes=state.state_bytes, state_bytes_gpu=state.state_bytes_on(device),
                         update_sec=0.0)
    if args.arm == "M_coupled":
        # Borrow the existing CPU masters in physical parameter order, after
        # synchronous training updates. A checkpoint would copy the whole state.
        fp32_master_cpu = [entry.master for entry in state._entries]
    master_idle = {"reverse_flip_count": 0, "coo_bytes_raw": 0} if master_arm else {}
    training_peak_bytes = 0

    rng_sched = np.random.default_rng(args.seed)
    step_logs, checkpoints = [], []
    effective_steps = 0
    total_gen_tokens = 0
    timing_totals = {"rollout_sec": 0.0, "grade_sec": 0.0, "backward_sec": 0.0, "update_sec": 0.0}

    for st in range(1, args.steps + 1):
        t_st = time.perf_counter()
        idxs = rng_sched.choice(len(train_items), size=args.batch_prompts, replace=False)
        b_items = [train_items[int(i)] for i in idxs]
        prompts = [format_prompt(it["problem"]) for it in b_items]

        torch.cuda.synchronize()
        t_r0 = time.perf_counter()
        outs = matched_generate_batch(
            model, tokenizer, prompts, device,
            max_new_tokens=args.max_tokens, n_samples=args.G, temperature=1.0,
            seed=args.seed * 10000 + st, batch_size=args.batch_prompts * args.G
        )
        torch.cuda.synchronize()
        timing_totals["rollout_sec"] += time.perf_counter() - t_r0

        t_g0 = time.perf_counter()
        preds = [txt for o in outs for txt in o["texts"]]
        golds = [it["answer"] for it in b_items for _ in range(args.G)]
        flags = math_grade(preds, golds)
        timing_totals["grade_sec"] += time.perf_counter() - t_g0

        b_recs = []
        for i, o in enumerate(outs):
            rws = [1.0 if flags[i * args.G + gi] else 0.0 for gi in range(args.G)]
            for c in o["comp_ids"]:
                total_gen_tokens += len(c)
            mr = sum(rws) / args.G
            advs = [r - mr for r in rws]
            b_recs.append({"prompt_ids": o["prompt_ids"], "comp_ids": o["comp_ids"], "rewards": rws, "advs": advs, "all_same": all(abs(a) < 1e-12 for a in advs)})

        mean_rw = float(np.mean([r for rec in b_recs for r in rec["rewards"]]))
        mixed_frac = float(np.mean([not rec["all_same"] for rec in b_recs]))
        raw_norm, chg_frac, did = 0.0, 0.0, False
        update_log = {**mcsr_idle, **master_idle}
        if mixed_frac > 0.0:
            raw_norm, chg_frac, did, t_bwd, t_upd, counts = apply_matched_step(
                args.arm, model, b_recs, fp32_master_cpu, args.eta, args.seed, st,
                state=state, adam_state=adam_state, master_telemetry=master_arm,
            )
            update_log.update(counts)
            if state is not None:
                update_log["update_sec"] = t_upd
            timing_totals["backward_sec"] += t_bwd
            timing_totals["update_sec"] += t_upd
            if did:
                effective_steps += 1

        step_log = {"step": st, "mean_rw": mean_rw, "mixed_frac": mixed_frac, "raw_norm": raw_norm,
                    "step_changed_frac": chg_frac, "did_update": did,
                    "step_sec": time.perf_counter() - t_st, **update_log}
        if master_arm and (st % 20 == 0 or st == args.steps):
            step_log["sticking_frac"] = sticking_fraction(ps, fp32_master_cpu, anchor_bf16_cpu)
        step_logs.append(step_log)

        if st % 40 == 0 or st == args.steps:
            kl_ref = compute_kl_from_ref(model, ref_h_cpu, ref_hw_bf16_cpu, kl_prefix_recs, device)
            dev_ev = None if skip_math else evaluate_suite(
                model, tokenizer, dev_val_items, math_grade, device, args.max_tokens, 0, args.seed, args.eval_bs
            )
            ck = {
                "step": st,
                "dev40_greedy": None if dev_ev is None else dev_ev["greedy_pass1"],
                "kl_from_ref": kl_ref,
                "recent40_rw": float(np.mean([x["mean_rw"] for x in step_logs[-40:]])),
                "recent40_chg_frac": float(np.mean([x["step_changed_frac"] for x in step_logs[-40:]])),
                "elapsed_sec": time.perf_counter() - t_start,
            }
            if master_arm:
                training_peak_bytes = max(training_peak_bytes, torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
                serving_logps = reference_token_logps(model, kl_prefix_recs, device)
                shadow = fp32_eval_model(args.model, device)
                load_fp32_master(shadow, fp32_master_cpu)
                master_logps = reference_token_logps(shadow, kl_prefix_recs, device)
                lag = (master_logps - serving_logps).abs().mean().item() if master_logps.numel() else None
                ck["hidden_lag_mean_abs_dlogp"] = step_log["hidden_lag_mean_abs_dlogp"] = lag
                ck["hidden_lag_peak_vram_GiB"] = torch.cuda.max_memory_allocated() / 1024**3
                del shadow
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            checkpoints.append(ck)
            dev_label = "skipped" if dev_ev is None else f"{ck['dev40_greedy']:.3f}"
            print(f"[step {st:3d}/{args.steps}] dev40={dev_label} rw40={ck['recent40_rw']:.3f} KL={kl_ref:.4e} chg={ck['recent40_chg_frac']:.5f} elapsed={ck['elapsed_sec']:.1f}s", flush=True)

    # Final Step-200 multi-domain evaluation
    gsm_diag = {}
    evals = final_suites(model, completions_dir, diag=gsm_diag)
    wt_end = compute_wikitext_ppl(model, wt_ids, device)
    kl_end = compute_kl_from_ref(model, ref_h_cpu, ref_hw_bf16_cpu, kl_prefix_recs, device)

    # ATOMIC SAVE BEFORE FROZEN MASTER AUDIT (sol P1: never lose main results if audit fails!)
    res = {
        "config": config,
        "core_manifest": core_manifest,
        "protocol": protocol,
        "kl_prefix_sha256": kl_prefix_sha256,
        "n_params": n_params,
        "effective_steps": effective_steps,
        "total_gen_tokens": total_gen_tokens,
        "math_grader_errors": math_err_cnt[0],
        "timing_totals": timing_totals,
        "peak_vram_GiB": max(training_peak_bytes, torch.cuda.max_memory_allocated()) / 1024**3,
        "elapsed_sec": time.perf_counter() - t_start,
        "wikitext2_step0": wt0,
        "wikitext2_step200": wt_end,
        "kl_step200": kl_end,
        "checkpoints": checkpoints,
        **evals,
        "master_audit": None,
        "step_logs": step_logs,
    }
    if completions_dir is not None:
        res["gsm8k_grader_diag"] = gsm_diag
    atomic_write_json(args.out, res)
    if ckpt_path:
        save_final_checkpoint(ckpt_path, model, ps, state, fp32_master_cpu, config, kl_prefix_sha256, args.steps)
    print(f"[SAVED main arm={args.arm}] {scores_label(evals)} | WT2 PPL={wt_end['ppl']:.4f} | KL={kl_end:.4e} | peak_vram={res['peak_vram_GiB']:.2f}GiB", flush=True)
    del adam_state

    def save_master_end_eval(model_fp32):
        directory = None if completions_dir is None else completions_dir / "master_end_fp32"
        res["master_end_eval"] = {
            "weights_dtype": "float32", "backbone_dtype": "float32", "head_dtype": "float32",
            **final_suites(model_fp32, directory, n_samples=8, batch_size=min(args.eval_bs, 16)),
        }
        res["elapsed_sec"] = time.perf_counter() - t_start
        atomic_write_json(args.out, res)

    faircast_audit = args.arm in ("M_faircast", "AdamW_faircast_sr")
    if eval_master_end and not faircast_audit:
        del state, ps, model
        torch.cuda.empty_cache()
        model_fp32 = fp32_eval_model(args.model, device)
        load_fp32_master(model_fp32, fp32_master_cpu)
        save_master_end_eval(model_fp32)

    # Clean Frozen Step-200 Master Audit (astra Req #2 + sol P1 clean same-FP32-backbone reference!)
    if faircast_audit:
        # 1) BF16 deployment RTNE export (eval MATH + GSM8K + WT2 + KL vs BF16 anchor)
        rtne_bf16_cpu = [m.bfloat16().clone() for m in fp32_master_cpu]
        with torch.no_grad():
            for p, w_rt in zip(ps, rtne_bf16_cpu):
                p.data.copy_(w_rt.to(device))
        rtne_kl_bf16 = compute_kl_from_ref(model, ref_h_cpu, ref_hw_bf16_cpu, kl_prefix_recs, device)
        rtne_wt_bf16 = compute_wikitext_ppl(model, wt_ids, device)
        rtne_dir = None if completions_dir is None else completions_dir / "master_audit_rtne_bf16"
        rtne_evals = final_suites(model, rtne_dir)
        rtne_math, rtne_gsm = rtne_evals["math500_test"], rtne_evals["gsm8k_ood"]

        # 2) 5 independent BF16 SR re-exports + theoretical 2p(1-p) check (separating saturated/nonfinite)
        theo_sum = 0.0
        off_grid_cnt = 0
        sat_cnt = 0
        for m_cpu in fp32_master_cpu:
            mf = m_cpu.to(device).view(-1)
            for _, blk in blocks(mf, 1 << 20):
                bits = blk.view(torch.int32)
                mag = (bits >> 16) & 32767
                sat_mask = mag >= 0x7F7F
                sat_cnt += int(sat_mask.sum().item())
                low = (bits & 65535).double() / 65536.0
                valid = ~sat_mask
                off_grid_cnt += int(((low > 0) & valid).sum().item())
                theo_sum += float((2.0 * low * (1.0 - low) * valid.double()).sum().item())
        theo_flip_rate = theo_sum / n_params

        sr_kls_bf16, sr_ppls_bf16, adj_flips = [], [], []
        sr_exports_cpu = []
        for r_i in range(5):
            cur_export_cpu = []
            diff_cnt = 0
            for j, (p, m_cpu) in enumerate(zip(ps, fp32_master_cpu)):
                gen = torch.Generator(device=device).manual_seed(900001 + r_i * 10007 + j)
                m_gpu = m_cpu.to(device)
                sr_into(m_gpu, p.data, gen)
                if sr_exports_cpu:
                    diff_cnt += int((p.data.view(torch.int16) != sr_exports_cpu[-1][j].to(device).view(torch.int16)).sum().item())
                cur_export_cpu.append(p.data.detach().cpu().clone())
            if sr_exports_cpu:
                adj_flips.append(diff_cnt / n_params)
            sr_exports_cpu.append(cur_export_cpu)
            sr_kls_bf16.append(compute_kl_from_ref(model, ref_h_cpu, ref_hw_bf16_cpu, kl_prefix_recs, device))
            sr_ppls_bf16.append(compute_wikitext_ppl(model, wt_ids, device)["ppl"])

        # 3) Clean SAME-FP32-BACKBONE reference audit (sol P1 fix!):
        #    Upcast Step-0 BF16 anchor -> FP32 backbone to measure:
        #    (a) Step-0 BF16-vs-FP32 backbone precision offset;
        #    (b) Exact FP32 master KL vs FP32 anchor;
        #    (c) Upcast RTNE & upcast 5x SR exports KL vs FP32 anchor AND KL directly vs exact FP32 master!
        del p, ps, model
        torch.cuda.empty_cache()
        model_fp32 = fp32_eval_model(args.model, device)
        ps32 = get_unique_params(model_fp32)

        # Load Step-0 BF16 anchor upcast to FP32
        with torch.no_grad():
            for p32, w0 in zip(ps32, anchor_bf16_cpu):
                p32.data.copy_(w0.float().to(device))
        fp32_anchor_vs_bf16_anchor_kl = compute_kl_from_ref(model_fp32, ref_h_cpu, ref_hw_bf16_cpu, kl_prefix_recs, device)
        ref32_h_cpu, ref32_hw_cpu = cache_anchor_kl_prefix(model_fp32, kl_prefix_recs, device)

        # Load exact Step-200 FP32 master
        load_fp32_master(model_fp32, fp32_master_cpu)
        if eval_master_end:
            save_master_end_eval(model_fp32)
        fp32_master_kl_vs_fp32_anchor = compute_kl_from_ref(model_fp32, ref32_h_cpu, ref32_hw_cpu, kl_prefix_recs, device)
        fp32_master_wt = compute_wikitext_ppl(model_fp32, wt_ids, device)
        m32_h_cpu, m32_hw_cpu = cache_anchor_kl_prefix(model_fp32, kl_prefix_recs, device)

        # Measure upcast RTNE export on identical FP32 backbone (vs FP32 anchor AND directly vs FP32 master!)
        with torch.no_grad():
            for p32, w_rt in zip(ps32, rtne_bf16_cpu):
                p32.data.copy_(w_rt.float().to(device))
        rtne_kl_vs_fp32_anchor = compute_kl_from_ref(model_fp32, ref32_h_cpu, ref32_hw_cpu, kl_prefix_recs, device)
        rtne_kl_vs_fp32_master = compute_kl_from_ref(model_fp32, m32_h_cpu, m32_hw_cpu, kl_prefix_recs, device)

        # Measure upcast 5x SR exports on identical FP32 backbone (vs FP32 anchor AND directly vs FP32 master!)
        sr_kls_vs_fp32_anchor, sr_kls_vs_fp32_master = [], []
        for cur_exp in sr_exports_cpu:
            with torch.no_grad():
                for p32, w_sr in zip(ps32, cur_exp):
                    p32.data.copy_(w_sr.float().to(device))
            sr_kls_vs_fp32_anchor.append(compute_kl_from_ref(model_fp32, ref32_h_cpu, ref32_hw_cpu, kl_prefix_recs, device))
            sr_kls_vs_fp32_master.append(compute_kl_from_ref(model_fp32, m32_h_cpu, m32_hw_cpu, kl_prefix_recs, device))

        del ps32, model_fp32
        torch.cuda.empty_cache()

        res["master_audit"] = {
            "off_grid_frac": off_grid_cnt / n_params,
            "saturated_or_nonfinite_cnt": sat_cnt,
            "theoretical_2p1mp": theo_flip_rate,
            "empirical_adj_flips_4pairs": adj_flips,
            "empirical_adj_flip_mean": float(np.mean(adj_flips)),
            "bf16_deployment": {
                "rtne_kl_vs_bf16_anchor": rtne_kl_bf16,
                "rtne_wikitext2": rtne_wt_bf16,
                "rtne_math500_test": rtne_math,
                "rtne_gsm8k_ood": rtne_gsm,
                "sr5_kls_vs_bf16_anchor": sr_kls_bf16,
                "sr5_kl_mean": float(np.mean(sr_kls_bf16)),
                "sr5_kl_std": float(np.std(sr_kls_bf16)),
                "sr5_ppls": sr_ppls_bf16,
                "sr5_ppl_mean": float(np.mean(sr_ppls_bf16)),
            },
            "same_fp32_backbone": {
                "step0_fp32_vs_bf16_backbone_kl": fp32_anchor_vs_bf16_anchor_kl,
                "exact_fp32_master_kl_vs_fp32_anchor": fp32_master_kl_vs_fp32_anchor,
                "exact_fp32_master_wikitext2": fp32_master_wt,
                "upcast_rtne_kl_vs_fp32_anchor": rtne_kl_vs_fp32_anchor,
                "upcast_rtne_kl_vs_fp32_master": rtne_kl_vs_fp32_master,
                "upcast_sr5_kls_vs_fp32_anchor": sr_kls_vs_fp32_anchor,
                "upcast_sr5_kl_vs_fp32_anchor_mean": float(np.mean(sr_kls_vs_fp32_anchor)),
                "upcast_sr5_kls_vs_fp32_master": sr_kls_vs_fp32_master,
                "upcast_sr5_kl_vs_fp32_master_mean": float(np.mean(sr_kls_vs_fp32_master)),
            },
        }
        res["elapsed_sec"] = time.perf_counter() - t_start
        atomic_write_json(args.out, res)
        print(
            f"[MASTER AUDIT CLEAN] step0_offset={fp32_anchor_vs_bf16_anchor_kl:.4e} | "
            f"FP32_master_KL={fp32_master_kl_vs_fp32_anchor:.4e} (PPL={fp32_master_wt['ppl']:.4f}) | "
            f"RTNE_vs_master_KL={rtne_kl_vs_fp32_master:.4e} ({scores_label(rtne_evals)}) | "
            f"SR5_vs_master_KL={np.mean(sr_kls_vs_fp32_master):.4e} | flip={np.mean(adj_flips)*100:.5f}% vs theo={theo_flip_rate*100:.5f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
