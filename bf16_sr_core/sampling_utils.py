# sampling_utils.py — Matched FP32-head PyTorch rollout (top_k=0) & per-item keyed vLLM sampling
from unittest.mock import patch
import torch
import torch.nn.functional as F
from transformers import GenerationConfig


@torch.inference_mode()
def generate_fp32_head(model, ids: torch.Tensor, mask: torch.Tensor, tokenizer, limit: int, temperature: float, seed: int):
    model.gradient_checkpointing_disable()
    model.eval()
    head = model.get_output_embeddings()
    weight = head.weight.detach().float()
    bias = None if head.bias is None else head.bias.detach().float()
    sampling = temperature > 0
    extra = dict(top_k=0, top_p=1.0, temperature=temperature) if sampling else {}
    cfg = GenerationConfig(
        do_sample=sampling,
        max_new_tokens=limit,
        num_beams=1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        **extra,
    )
    dev = ids.device.index
    if dev is None:
        dev = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[dev]):
        torch.cuda.default_generators[dev].manual_seed(seed)
        with patch.object(head, "forward", lambda h: F.linear(h.float(), weight, bias)):
            return model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)


def request_seed(phase: int, run_seed: int, step: int, item_id: int, n: int) -> int:
    if not (0 <= phase < 8 and 0 <= run_seed < 65536 and 0 <= step < 65536 and 0 <= item_id < 16384 and 1 <= n <= 256):
        raise ValueError("seed fields out of range")
    return (((((phase << 16) | run_seed) << 16 | step) << 14 | item_id) << 8)


def generate_vllm_keyed(llm, prompts, item_ids, phase: int, run_seed: int, step: int, n: int, temperature: float, limit: int):
    from vllm import SamplingParams

    params = [
        SamplingParams(
            n=n,
            seed=request_seed(phase, run_seed, step, i, n),
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            max_tokens=limit,
        )
        for i in item_ids
    ]
    return llm.generate(prompts, params)
