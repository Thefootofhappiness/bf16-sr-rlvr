#!/usr/bin/env python3
"""Wave 2B (v2.1, sol-audited): Procedural Countdown-4 Arithmetic RLVR with strict AST verifier & bf16_sr_core.

Fixes applied from gpt-6.2-sol audit (2026-11-20):
- P0: Removed @torch.no_grad() from apply_step; forward + hidden slice inside torch.enable_grad() + assert h.requires_grad.
- P0: Eliminated reward hacking in safe_eval_countdown: NO arbitrary char stripping; balanced-brace ONLY LAST \\boxed{};
      explicit \\frac{a}{b} conversion; at most one '=' requiring exact target on RHS; strict fullmatch before AST eval.
- P1: Disjoint digit-multiset grouping (tuple(sorted(nums))) before train/dev/kl/test splitting (zero multiset leakage).
- P1: Prompt explicitly states rational/negative intermediate values are allowed and asks for concise derivation.
- P1: Default max_tokens=512 (with truncation & boxed-rate diagnostics saved at every checkpoint).
- P1: Identical FP32 update arithmetic d = gb.float().mul_(alpha); target.add_(d) across all 4 arms.
"""
import argparse
import ast
import hashlib
import json
import math
import operator
import os
import re
import sys
import time
from collections import Counter
from fractions import Fraction

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
import sampling_utils
import sr_step
from bf16_sr import sr_into
from sampling_utils import generate_fp32_head
from sr_step import StepGrads, blocks, grad_sqnorm, head_tiles

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def generate_disjoint_countdown_splits(seed: int = 20261120):
    """Generate 492 Countdown-3 puzzles (3 integers in [1,18], exact integer intermediates, target in [5,60])
    with strictly disjoint sorted digit multisets across train(240)/dev(40)/kl(12)/test(200)."""
    rng = np.random.default_rng(seed)
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
        ("/", lambda a, b: a // b if b != 0 and a % b == 0 else None),
    ]
    seen_multisets = set()
    puzzles = []
    n_needed = 240 + 40 + 12 + 200  # 492 unique 3-number multisets out of C(20,3)=1140 -> zero split overlap
    while len(puzzles) < n_needed:
        nums = rng.integers(1, 19, size=3).tolist()
        mkey = tuple(sorted(nums))
        if mkey in seen_multisets:
            continue
        perm = rng.permutation(nums).tolist()
        vals = list(perm)
        exprs = [str(x) for x in perm]
        ok = True
        while len(vals) > 1:
            i = int(rng.integers(0, len(vals) - 1))
            op_sym, op_fn = ops[int(rng.integers(0, len(ops)))]
            res = op_fn(vals[i], vals[i + 1])
            if res is None or res < 0 or res > 200:
                ok = False
                break
            new_e = f"({exprs[i]} {op_sym} {exprs[i + 1]})"
            vals = vals[:i] + [res] + vals[i + 2 :]
            exprs = exprs[:i] + [new_e] + exprs[i + 2 :]
        if not ok:
            continue
        t_int = int(vals[0])
        if not (5 <= t_int <= 60):
            continue
        seen_multisets.add(mkey)
        puzzles.append({"nums": nums, "target": t_int, "ref_expr": exprs[0]})
    return puzzles[:240], puzzles[240:280], puzzles[280:292], puzzles[292:492]


def format_countdown_prompt(item: dict) -> str:
    nums_str = ", ".join(str(x) for x in item["nums"])
    return (
        f"Using the numbers [{nums_str}], build an arithmetic expression that equals {item['target']}. "
        f"Use each given number exactly once with only +, -, *, /, and parentheses "
        f"(fractional or negative intermediate values are allowed). "
        f"Keep your reasoning concise, and put ONLY the final expression inside a single \\boxed{{}}.\n\nSolution:"
    )


def extract_last_boxed(text: str):
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
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
    return "".join(out).strip() if depth == 0 else None


def _replace_latex_frac(s: str) -> str:
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


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_eval_countdown(expr_str: str, expected_nums: list, target: int) -> bool:
    if expr_str is None:
        return False
    s = expr_str.strip()
    if not s or len(s) > 120:
        return False
    # Normalize ONLY explicit supported LaTeX/Unicode arithmetic operators (never strip unknown chars!)
    s = s.replace("\\left(", "(").replace("\\right)", ")").replace("\\left[", "(").replace("\\right]", ")")
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    s = _replace_latex_frac(s)

    # If '=' appears, allow AT MOST ONE '=' and require RHS == exact target integer!
    if "=" in s:
        parts = s.split("=")
        if len(parts) != 2:
            return False
        lhs, rhs = parts[0].strip(), parts[1].strip()
        if not re.fullmatch(r"[-+]?\d+", rhs) or int(rhs) != target:
            return False
        s = lhs

    # Strict fullmatch: reject decimals (2.5), powers (^), letters, or arbitrary punctuation!
    if not re.fullmatch(r"[0-9\+\-\*/\(\)\s]+", s):
        return False
    if "//" in s or "**" in s:
        return False
    try:
        tree = ast.parse(s, mode="eval")
    except Exception:
        return False

    used_nums = []

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            used_nums.append(int(node.value))
            return Fraction(int(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            lv = _eval(node.left)
            rv = _eval(node.right)
            if isinstance(node.op, ast.Div) and rv == 0:
                raise ZeroDivisionError()
            return _BIN_OPS[type(node.op)](lv, rv)
        raise ValueError("disallowed AST node")

    try:
        val = _eval(tree)
    except Exception:
        return False
    if Counter(used_nums) != Counter(expected_nums):
        return False
    return val == Fraction(target)


def grade_countdown_batch(preds, items):
    """Grade ONLY the last complete \\boxed{...}."""
    out = []
    has_box = 0
    for text, it in zip(preds, items):
        cand = extract_last_boxed(text)
        if cand is not None:
            has_box += 1
        out.append(safe_eval_countdown(cand, it["nums"], it["target"]))
    return out, has_box / max(len(preds), 1)


def get_unique_params(model):
    ps = [p for p in model.parameters() if p.requires_grad]
    ptrs = [p.untyped_storage().data_ptr() for p in ps]
    if len(set(ptrs)) != len(ptrs):
        raise ValueError("shared storage across distinct Parameter objects")
    return ps


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
        out_ids = generate_fp32_head(model, input_ids, attn_mask, tokenizer, limit=max_new_tokens, temperature=temperature, seed=seed + s)
        for seq in out_ids[:, in_len:].cpu().tolist():
            if eos_id in seq:
                comp = seq[: seq.index(eos_id) + 1]
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


def apply_step(arm: str, model, records, fp32_master_cpu, eta: float, seed: int, step: int, rows: int = 2048, chunk: int = 1 << 20):
    """NOTE: Do NOT decorate with @torch.no_grad(); forward + hidden slice MUST run with autograd enabled."""
    model.train()
    model.gradient_checkpointing_enable()
    w_head = model.get_output_embeddings().weight
    tied = w_head is model.get_input_embeddings().weight
    nt = max(sum(len(c) for rec in records for c in rec["comp_ids"]), 1)

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
                assert h.requires_grad and h.shape[0] == len(c)
                head_tiles(h, w_head, ids[len(prompt) :], adv / nt, sink, tied, rows)

        with torch.no_grad():
            ps = sink.ps
            gs = [p.grad for p in ps]
            squares = [grad_sqnorm(g, chunk) for g in gs if g is not None]
            if not squares:
                return 0.0, 0.0, False
            norm = torch.stack(squares).sum().sqrt().item()
            if not math.isfinite(norm):
                raise FloatingPointError(f"nonfinite norm at step {step}")
            if norm == 0.0:
                for p in ps:
                    p.grad = None
                return 0.0, 0.0, False

            alpha = -eta * min(1.0, 1.0 / norm)
            dev = ps[0].device
            changed_tot = torch.zeros((), device=dev, dtype=torch.int64)
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
                elif arm in ("M_faircast", "M_rtne"):
                    m_cpu = fp32_master_cpu[j].view(-1)
                    for i, gb in blocks(g, chunk):
                        old = w[i : i + gb.numel()]
                        d = gb.float().mul_(alpha)
                        m_blk = m_cpu[i : i + gb.numel()].to(dev)
                        m_blk.add_(d)
                        del d
                        m_cpu[i : i + gb.numel()].copy_(m_blk.cpu())
                        if arm == "M_rtne":
                            dst = m_blk.bfloat16()
                        else:
                            dst = torch.empty_like(old)
                            sr_into(m_blk, dst, gen, chunk)
                        changed_tot.add_(torch.count_nonzero(dst.view(torch.int16) != old.view(torch.int16)))
                        old.copy_(dst)
                elif arm == "R_pure_bf16":
                    for i, gb in blocks(g, chunk):
                        old = w[i : i + gb.numel()]
                        d = gb.float().mul_(alpha)
                        dst = old.float().add_(d).bfloat16()
                        del d
                        changed_tot.add_(torch.count_nonzero(dst.view(torch.int16) != old.view(torch.int16)))
                        old.copy_(dst)
                p.grad = None
            return norm, changed_tot.item() / max(n_params, 1), True


@torch.inference_mode()
def cache_anchor_kl(model, kl_recs, device):
    model.eval()
    ref_h_cpu = []
    for rec in kl_recs:
        Lp = len(rec["prompt_ids"])
        for c in rec["comp_ids"]:
            if not c:
                continue
            full = torch.tensor([rec["prompt_ids"] + c], dtype=torch.long, device=device)
            ref_h_cpu.append(model.model(input_ids=full, use_cache=False).last_hidden_state[0, Lp - 1 : -1].detach().cpu())
    return ref_h_cpu, model.get_output_embeddings().weight.detach().cpu().clone()


@torch.inference_mode()
def compute_kl(model, ref_h_cpu, ref_hw_cpu, kl_recs, device):
    model.eval()
    ref_hw = ref_hw_cpu.to(device).float()
    cur_hw = model.get_output_embeddings().weight.detach().float()
    tot_kl, tot_tok, idx = 0.0, 0, 0
    for rec in kl_recs:
        Lp = len(rec["prompt_ids"])
        for c in rec["comp_ids"]:
            if not c:
                continue
            full = torch.tensor([rec["prompt_ids"] + c], dtype=torch.long, device=device)
            cur_h = model.model(input_ids=full, use_cache=False).last_hidden_state[0, Lp - 1 : -1].float()
            ref_h = ref_h_cpu[idx].to(device).float()
            idx += 1
            for s in range(0, cur_h.shape[0], 64):
                e = min(s + 64, cur_h.shape[0])
                lp_r = F.log_softmax(F.linear(ref_h[s:e], ref_hw), dim=-1)
                lp_c = F.log_softmax(F.linear(cur_h[s:e], cur_hw), dim=-1)
                tot_kl += float((lp_r.exp() * (lp_r - lp_c)).sum().item())
                tot_tok += e - s
    return tot_kl / max(tot_tok, 1)


def evaluate_countdown(model, tokenizer, items, device, max_tokens: int, n_samples: int, seed: int, batch_size: int = 64):
    prompts = [format_countdown_prompt(it) for it in items]
    t0 = time.perf_counter()
    outs_g = matched_generate_batch(model, tokenizer, prompts, device, max_tokens, 1, 0.0, seed, batch_size)
    flags_g, box_rate_g = grade_countdown_batch([o["texts"][0] for o in outs_g], items)
    trunc_rate_g = float(np.mean([1 if o["truncs"][0] else 0 for o in outs_g]))
    per_item_p1 = []
    box_rate_s, trunc_rate_s = 0.0, 0.0
    if n_samples > 0:
        outs_s = matched_generate_batch(model, tokenizer, prompts, device, max_tokens, n_samples, 0.6, seed + 100000, batch_size)
        preds_s = [txt for o in outs_s for txt in o["texts"]]
        items_s = [it for it in items for _ in range(n_samples)]
        flags_s, box_rate_s = grade_countdown_batch(preds_s, items_s)
        trunc_rate_s = float(np.mean([1 if tr else 0 for o in outs_s for tr in o["truncs"]]))
        for i in range(len(items)):
            per_item_p1.append(sum(1 for x in flags_s[i * n_samples : (i + 1) * n_samples] if x) / n_samples)
    return {
        "n_items": len(items),
        "n_samples": n_samples,
        "greedy_pass1": float(np.mean(flags_g)),
        "greedy_box_rate": box_rate_g,
        "greedy_trunc_rate": trunc_rate_g,
        "stoch_pass1": float(np.mean(per_item_p1)) if per_item_p1 else None,
        "stoch_box_rate": box_rate_s,
        "stoch_trunc_rate": trunc_rate_s,
        "eval_sec": time.perf_counter() - t0,
        "per_item_greedy": [1 if x else 0 for x in flags_g],
        "per_item_p1": per_item_p1,
    }


def atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["anchor_only", "S_full_sr", "M_faircast", "M_rtne", "R_pure_bf16"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--eval-n", type=int, default=8)
    ap.add_argument("--eval-bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.perf_counter()
    device = torch.device("cuda:0")
    train_items, dev_items, kl_items, test_items = generate_disjoint_countdown_splits(seed=20261120)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    ps = get_unique_params(model)
    anchor_cpu = [p.data.detach().cpu().clone() for p in ps]
    fp32_master_cpu = [w.float().clone() for w in anchor_cpu] if args.arm in ("M_faircast", "M_rtne") else None

    kl_outs = matched_generate_batch(model, tokenizer, [format_countdown_prompt(it) for it in kl_items], device, 384, 2, 1.0, 9999, 24)
    kl_recs = [{"prompt_ids": o["prompt_ids"], "comp_ids": o["comp_ids"]} for o in kl_outs]
    kl_sha = hashlib.sha256(json.dumps(kl_recs).encode()).hexdigest()
    ref_h_cpu, ref_hw_cpu = cache_anchor_kl(model, kl_recs, device)

    dev0 = evaluate_countdown(model, tokenizer, dev_items, device, args.max_tokens, 0, args.seed, args.eval_bs)
    print(f"[start countdown_v2.1 arm={args.arm} eta={args.eta} seed={args.seed}] dev40_greedy={dev0['greedy_pass1']:.4f} box={dev0['greedy_box_rate']:.2f} trunc={dev0['greedy_trunc_rate']:.2f} kl_sha={kl_sha[:12]}", flush=True)

    if args.arm == "anchor_only":
        te = evaluate_countdown(model, tokenizer, test_items, device, args.max_tokens, args.eval_n, args.seed + 777, args.eval_bs)
        res = {"config": vars(args), "protocol": "matched_fp32_head_topk0_countdown4_v2.1", "kl_sha": kl_sha, "dev40_step0": dev0, "test_eval": te}
        atomic_write_json(args.out, res)
        print(f"[DONE countdown anchor] greedy={te['greedy_pass1']:.4f} stoch={te['stoch_pass1']:.4f} box={te['stoch_box_rate']:.2f} trunc={te['stoch_trunc_rate']:.2f}", flush=True)
        return

    rng_sched = np.random.default_rng(args.seed)
    step_logs, checkpoints = [], []
    effective_steps = 0

    for st in range(1, args.steps + 1):
        idxs = rng_sched.choice(len(train_items), size=args.batch_prompts, replace=False)
        b_items = [train_items[int(i)] for i in idxs]
        outs = matched_generate_batch(model, tokenizer, [format_countdown_prompt(it) for it in b_items], device, args.max_tokens, args.G, 1.0, args.seed * 10000 + st, args.batch_prompts * args.G)
        preds = [txt for o in outs for txt in o["texts"]]
        items_rep = [it for it in b_items for _ in range(args.G)]
        flags, box_rate = grade_countdown_batch(preds, items_rep)
        trunc_rate = float(np.mean([1 if tr else 0 for o in outs for tr in o["truncs"]]))

        b_recs = []
        for i, o in enumerate(outs):
            rws = [1.0 if flags[i * args.G + gi] else 0.0 for gi in range(args.G)]
            mr = sum(rws) / args.G
            advs = [r - mr for r in rws]
            b_recs.append({"prompt_ids": o["prompt_ids"], "comp_ids": o["comp_ids"], "rewards": rws, "advs": advs, "all_same": all(abs(a) < 1e-12 for a in advs)})

        mean_rw = float(np.mean([r for rec in b_recs for r in rec["rewards"]]))
        mixed_frac = float(np.mean([not rec["all_same"] for rec in b_recs]))
        raw_norm, chg_frac, did = 0.0, 0.0, False
        if mixed_frac > 0.0:
            raw_norm, chg_frac, did = apply_step(args.arm, model, b_recs, fp32_master_cpu, args.eta, args.seed, st)
            if did:
                effective_steps += 1
        step_logs.append({"step": st, "mean_rw": mean_rw, "mixed_frac": mixed_frac, "box_rate": box_rate, "trunc_rate": trunc_rate, "raw_norm": raw_norm, "step_changed_frac": chg_frac, "did_update": did})

        if st % 40 == 0 or st == args.steps:
            kl_ref = compute_kl(model, ref_h_cpu, ref_hw_cpu, kl_recs, device)
            dev_ev = evaluate_countdown(model, tokenizer, dev_items, device, args.max_tokens, 0, args.seed, args.eval_bs)
            ck = {
                "step": st,
                "dev40_greedy": dev_ev["greedy_pass1"],
                "kl_from_ref": kl_ref,
                "recent40_rw": float(np.mean([x["mean_rw"] for x in step_logs[-40:]])),
                "recent40_mixed": float(np.mean([x["mixed_frac"] for x in step_logs[-40:]])),
                "recent40_trunc": float(np.mean([x["trunc_rate"] for x in step_logs[-40:]])),
                "recent40_chg_frac": float(np.mean([x["step_changed_frac"] for x in step_logs[-40:]])),
                "elapsed_sec": time.perf_counter() - t_start,
            }
            checkpoints.append(ck)
            print(f"[cd step {st:3d}/{args.steps}] dev40={ck['dev40_greedy']:.3f} rw40={ck['recent40_rw']:.3f} mix={ck['recent40_mixed']:.2f} trunc={ck['recent40_trunc']:.2f} KL={kl_ref:.4e} chg={ck['recent40_chg_frac']:.5f} elapsed={ck['elapsed_sec']:.1f}s", flush=True)

    test_ev = evaluate_countdown(model, tokenizer, test_items, device, args.max_tokens, args.eval_n, args.seed + 777, args.eval_bs)
    kl_end = compute_kl(model, ref_h_cpu, ref_hw_cpu, kl_recs, device)

    res = {
        "config": vars(args),
        "protocol": "matched_fp32_head_topk0_countdown4_v2.1",
        "kl_sha": kl_sha,
        "effective_steps": effective_steps,
        "kl_step200": kl_end,
        "elapsed_sec": time.perf_counter() - t_start,
        "checkpoints": checkpoints,
        "test_eval": test_ev,
        "test_eval_master_rtne": None,
        "step_logs": step_logs,
    }
    atomic_write_json(args.out, res)

    if args.arm == "M_faircast" and fp32_master_cpu is not None:
        with torch.no_grad():
            for p, m_cpu in zip(ps, fp32_master_cpu):
                p.data.copy_(m_cpu.to(device).bfloat16())
        rtne_ev = evaluate_countdown(model, tokenizer, test_items, device, args.max_tokens, args.eval_n, args.seed + 777, args.eval_bs)
        rtne_ev["kl_from_ref"] = compute_kl(model, ref_h_cpu, ref_hw_cpu, kl_recs, device)
        res["test_eval_master_rtne"] = rtne_ev
        atomic_write_json(args.out, res)

    print(f"[DONE countdown_v2.1 arm={args.arm}] greedy={test_ev['greedy_pass1']:.4f} stoch={test_ev['stoch_pass1']:.4f} KL={kl_end:.4e} elapsed={res['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
