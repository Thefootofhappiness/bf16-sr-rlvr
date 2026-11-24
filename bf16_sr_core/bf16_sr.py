# bf16_sr.py — Bit-level unbiased BF16 stochastic rounding (PyTorch 2.2+ & 2.8+ compatible, no uint32)
import math
import torch


@torch.no_grad()
def _round_tile(x: torch.Tensor, dst: torch.Tensor, draw: torch.Tensor) -> None:
    bits = x.view(torch.int32)
    hi = bits >> 16
    low = bits & 65535
    mag = hi & 32767

    # draw is disposable. Carry increases magnitude for either sign (IEEE-754 sign-magnitude).
    draw.lt_(low)
    draw.masked_fill_(mag >= 0x7F7F, 0)
    hi.add_(draw)

    # A NaN with payload only in the discarded low 16 bits must remain a NaN (quiet NaN).
    lost_nan = (mag == 0x7F80) & (low != 0)
    draw.zero_().masked_fill_(lost_nan, 0x40)
    hi.bitwise_or_(draw)
    dst.view(torch.int16).copy_(hi)


@torch.no_grad()
def sr_into(x: torch.Tensor, dst: torch.Tensor, gen: torch.Generator, chunk: int = 1 << 20) -> None:
    if (
        x.dtype != torch.float32
        or dst.dtype != torch.bfloat16
        or x.shape != dst.shape
        or x.device != dst.device
        or not x.is_contiguous()
        or not dst.is_contiguous()
        or chunk < 1
    ):
        raise ValueError("expected contiguous FP32 input and matching BF16 output")
    if x.numel() and x.untyped_storage().data_ptr() == dst.untyped_storage().data_ptr():
        raise ValueError("input and output must use separate storage")

    a, b = x.view(-1), dst.view(-1)
    for i in range(0, a.numel(), chunk):
        t = a[i : i + chunk]
        r = torch.randint(65536, t.shape, device=t.device, dtype=torch.int32, generator=gen)
        _round_tile(t, b[i : i + t.numel()], r)


def sr_cast_tensor_gpu(x_fp32: torch.Tensor, rng_gen: torch.Generator, inf_pos=None, inf_neg=None) -> torch.Tensor:
    out = torch.empty_like(x_fp32, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
    sr_into(x_fp32.contiguous(), out, rng_gen)
    return out


def test(device: str) -> None:
    gen = torch.Generator(device=device).manual_seed(197)

    def floats(words):
        signed = [w if w < 2**31 else w - 2**32 for w in words]
        return torch.tensor(signed, device=device, dtype=torch.int32).view(torch.float32)

    def codes(y):
        return y.view(torch.int16).int() & 65535

    def legacy(x, u):
        r = x.bfloat16()
        up = torch.nextafter(r, torch.full_like(r, math.inf))
        dn = torch.nextafter(r, torch.full_like(r, -math.inf))
        le = r.float() <= x
        lo = torch.where(le, r, dn)
        hi = torch.where(le, up, r)
        span = hi.float() - lo.float()
        p = torch.where(span > 0, (x - lo.float()) / span.clamp_min(1e-45), 0.0)
        return torch.where(u < p, hi, lo)

    # 1. Every BF16 encoding, including signed zeros, infinities, and encoded NaNs.
    grid = torch.arange(65536, device=device, dtype=torch.int32)
    x = (grid << 16).view(torch.float32)
    y = torch.empty_like(x, dtype=torch.bfloat16)
    sr_into(x, y, gen, 8191)
    assert torch.equal(codes(y), grid)

    # 2. Symmetric finite saturation at +-B_max, NaN preservation, Inf preservation.
    x = floats([0x7F7FFFFF, 0xFF7FFFFF, 0x7F800001, 0xFF800001, 0x7FC12345, 0x7F800000, 0xFF800000])
    y = torch.empty_like(x, dtype=torch.bfloat16)
    sr_into(x, y, gen, 3)
    assert codes(y).tolist() == [0x7F7F, 0xFF7F, 0x7FC0, 0xFFC0, 0x7FC1, 0x7F80, 0xFF80]

    # 3. Exhaustive 65,536-draw verification across subnormals, binade boundaries, negatives, extremes.
    r = torch.arange(65536, device=device, dtype=torch.int32)
    for word in (
        1,
        0x80000001,
        0xFFFF,
        0x807FFFFF,
        0x007FFFFF,
        0x3F7FF001,
        0x3F800001,
        0x3F804000,
        0xBF804000,
        0xBF7FFFFF,
        0x7F7EFFFF,
        0xFF7EFFFF,
    ):
        x = floats([word]).repeat(65536)
        y = torch.empty_like(x, dtype=torch.bfloat16)
        _round_tile(x, y, r.clone())
        away = (word >> 16) + 1
        assert (codes(y) == away).sum().item() == (word & 65535)
        u = (65535 - r if word >> 31 else r).float() / 65536
        assert torch.equal(codes(y), codes(legacy(x, u))), hex(word)

    # 4. Monte Carlo unbiasedness, variance, and two-point law agreement.
    for word in (0x3F804000, 0xBF804000):
        n, p, h = 262144, 0.25, 2**-7
        x = floats([word]).repeat(n)
        y = torch.empty_like(x, dtype=torch.bfloat16)
        sr_into(x, y, gen)
        old = legacy(x, torch.rand(n, device=device, generator=gen))
        err = y.double() - x.double()
        se_frequency = math.sqrt(p * (1 - p) / n)
        assert abs(err.mean().item()) < 7 * h * se_frequency
        assert abs(err.var(unbiased=False).item() / h**2 - p * (1 - p)) < 7 * se_frequency
        away = (word >> 16) + 1
        f1 = (codes(y) == away).double().mean().item()
        f2 = (codes(old) == away).double().mean().item()
        assert abs(f1 - f2) < 7 * math.sqrt(2) * se_frequency

    assert torch.tensor(1 + 2**-26, dtype=torch.float32).item() == 1.0
    print("PASS bf16_sr", torch.__version__, device)


if __name__ == "__main__":
    test("cuda" if torch.cuda.is_available() else "cpu")
