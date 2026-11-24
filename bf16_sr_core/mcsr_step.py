"""In-place MCSR with an FP32 RN-even target and integer probability masses.

Service tensors are contiguous BF16. A finite residual is a signed offset from
the service endpoint, increasing in numerical order. A packed int4 tensor owns
both nibbles of each byte, including a zero high nibble at an odd-length tail.
State and service weights form one checkpoint. Calls through the state manager
are serialized across streams. External access to live weights and training
forward/backward must be ordered against updates by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import hashlib
import math
from threading import Lock

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

DEFAULT_CHUNK = 524_288
MASTER = math.inf
_BMAX = 0x7F7F
_MASK32 = (1 << 32) - 1
_SUM_STRIPES = 32
_SUM_EXPONENTS = 254
# Statistics occupy one device vector until the entire update is complete.
STAT_KEYS = ("bit_changed_count", "reverse_flip_count", "saturated_count", "swallowed_count")


def _key(seed, step, param_idx, stage):
    raw = f"{seed}/{step}/{param_idx}/{stage}".encode("ascii")
    seed64 = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")
    # Philox consumes the generator's key with explicit counters instead of
    # consuming launch-dependent offsets from its mutable state.
    return torch.Generator(device="cpu").manual_seed(seed64).initial_seed()


if triton is not None:
    @triton.jit
    def _philox_cuda(index, retry, low_key, high_key):
        a = index.to(tl.uint32)
        b = (index >> 32).to(tl.uint32)
        c = retry.to(tl.uint32)
        d = (retry >> 32).to(tl.uint32)
        k0 = low_key.to(tl.uint32)
        k1 = high_key.to(tl.uint32)
        for _ in tl.static_range(10):
            ap = a.to(tl.uint64) * 0xD2511F53
            cp = c.to(tl.uint64) * 0xCD9E8D57
            a = (cp >> 32).to(tl.uint32) ^ b ^ k0
            b = cp.to(tl.uint32)
            c = (ap >> 32).to(tl.uint32) ^ d ^ k1
            d = ap.to(tl.uint32)
            k0 = k0 + 0x9E3779B9
            k1 = k1 + 0xBB67AE85
        return a

    @triton.jit
    def _uniform_cuda(bounds, output, size, start, low_key, high_key, fixed: tl.constexpr, tile: tl.constexpr):
        lane = tl.program_id(0) * tile + tl.arange(0, tile)
        index = lane.to(tl.int64) + start
        retry = tl.full((tile,), 0, tl.int64)
        r = _philox_cuda(index, retry, low_key, high_key)
        if fixed:
            out = r & (fixed - 1)
        else:
            n = tl.load(bounds + lane, lane < size, other=1).to(tl.uint32)
            threshold = (tl.full((tile,), 0, tl.uint32) - n) % n
            rejected = r < threshold
            while tl.sum(rejected.to(tl.int32), 0) > 0:
                retry += 1
                draw = _philox_cuda(index, retry, low_key, high_key)
                r = tl.where(rejected, draw, r)
                rejected = r < threshold
            out = r % n
        tl.store(output + lane, out.to(tl.int32), lane < size)


def _philox_cpu(index, retry, key):
    a, b = index & _MASK32, index >> 32
    c, d = retry & _MASK32, retry >> 32
    k0, k1 = key & _MASK32, key >> 32
    for _ in range(10):
        ap, cp = a * 0xD2511F53, c * 0xCD9E8D57
        a = ((cp >> 32) & _MASK32) ^ b ^ k0
        b = cp & _MASK32
        c = ((ap >> 32) & _MASK32) ^ d ^ k1
        d = ap & _MASK32
        k0 = (k0 + 0x9E3779B9) & _MASK32
        k1 = (k1 + 0xBB67AE85) & _MASK32
    return a


def _uniform(like, start, key, *, fixed=0, bounds=None):
    """Uniform integer in each lane's range. A retry only advances that lane."""
    size = like.numel()
    if like.is_cuda:
        out = torch.empty_like(like, dtype=torch.int32)
        with torch.cuda.device(like.device):
            _uniform_cuda[(triton.cdiv(size, 256),)](
                out if bounds is None else bounds, out, size, start,
                key & _MASK32, key >> 32, fixed, 256,
            )
        return out
    index = torch.arange(start, start + size, dtype=torch.int64, device=like.device)
    r = _philox_cpu(index, torch.zeros_like(index), key)
    if fixed:
        return (r & (fixed - 1)).int()
    n = bounds.long()
    threshold = (1 << 32) % n
    rejected = r < threshold
    retry = 0
    while rejected.any():  # CPU only. CUDA performs this loop on device.
        retry += 1
        pending = index[rejected]
        r[rejected] = _philox_cpu(pending, torch.full_like(pending, retry), key)
        rejected = r < threshold
    return (r % n).int()


def _rank(word):
    """Signed numerical BF16 rank with both zero codes mapped to rank zero."""
    mag = word & 0x7FFF
    return torch.where(word < 0, -mag, mag)


def _code(rank, negative_zero):
    return torch.where((rank < 0) | ((rank == 0) & negative_zero),
                       (-rank) | 0x8000, rank)


def _unpack(packed, count):
    out = torch.empty(count, dtype=torch.int32, device=packed.device)
    out[::2] = packed & 15
    out[1::2] = packed[:count // 2] >> 4
    return out.sub_(16 * (out >= 8))


def _pack_into(packed, offset):
    packed.copy_((offset[::2] & 15).to(torch.uint8))
    packed[:offset.numel() // 2].bitwise_or_((offset[1::2] << 4).to(torch.uint8))


def _decode(word, offset, shift):
    rank = _rank(word)
    negative = (rank < 0) | ((rank == 0) & ((offset < 0) | ((offset == 0) & (word < 0))))
    directed_offset = torch.where(negative, -offset, offset)
    mag = rank.abs() - (directed_offset < 0).int()
    fraction = (directed_offset & ((1 << (16 - shift)) - 1)) << shift
    bits = (mag << 16) | fraction
    return (bits | (negative.int() << 31)).view(torch.float32)


def _mass(bits, shift):
    """Near/away ranks and away mass, with point support encoded as one rank."""
    near = _rank(bits >> 16)
    k = (bits & 65535) >> shift
    away = near + (1 - 2 * (bits < 0).int()) * (k != 0)
    return near, away, k


def _mass_at(at, near, away, k, total):
    return torch.where(at == near, total - k, 0) + torch.where(at == away, k, 0)


def _coupling_masses(old_rank, old_bits, new_bits, shift):
    total = 1 << (16 - shift)
    a, b, p = _mass(old_bits, shift)
    c, e, q = _mass(new_bits, shift)
    old_here = _mass_at(old_rank, a, b, p, total)
    common_here = torch.minimum(old_here, _mass_at(old_rank, c, e, q, total))
    remaining_near = (total - q - _mass_at(c, a, b, p, total)).clamp_min_(0)
    remaining_away = (q - _mass_at(e, a, b, p, total)).clamp_min_(0)
    return c, e, q, old_here, common_here, remaining_near, remaining_away


def _fine_bits(bits, shift, start, key):
    # The fine-grid integer and fractional indices are separate bit fields.
    # Carry out of the fractional field reaches the next BF16 code exactly.
    radix = 1 << shift
    rem = bits & (radix - 1)
    draw = _uniform(bits, start, key, fixed=radix)
    return (bits & -radix) + ((draw < rem).int() << shift)


def _finite_view(t, dtype, shape, devices):
    if (not isinstance(t, torch.Tensor) or t.layout != torch.strided
            or t.dtype not in dtype or t.shape != shape or t.device not in devices
            or not t.is_contiguous()):
        raise ValueError("unexpected state/update dtype, shape, layout or device")


def _ident(t):
    return t.device, t.untyped_storage().data_ptr()


def _state_size(t):
    return t.numel() * t.element_size()


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


@dataclass
class _Entry:
    weight: torch.Tensor
    mode: str
    residual: torch.Tensor | None = None
    master: torch.Tensor | None = None

    @property
    def shift(self):
        return {"int4": 13, "int8": 9}.get(self.mode, 0)


class MCSR:
    """State for an ordered collection of weights.

    L=1 is reserved for the existing S rule on the non-norm coordinates of
    S_norm_fp32. Exact physical ties are deduplicated. Other storage aliases
    are rejected. A CPU-offloaded CUDA master uses pinned memory and copies
    queued on the caller's current stream.
    """

    @torch.no_grad()
    def __init__(self, weights, L, *, norm_weights=(), norm_service="bf16",
                 master_device=None, chunk_size=DEFAULT_CHUNK):
        if L not in (1, 8, 128, MASTER) or type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("expected L in {1, 8, 128, inf} and a positive integer chunk size")
        if norm_service not in ("bf16", "fp32"):
            raise ValueError("norm_service must be bf16 or fp32")
        weights = tuple(weights)
        if not weights or not isinstance(weights[0], torch.Tensor):
            raise ValueError("a nonempty collection of BF16 weights is required")
        self._lock = Lock()
        self.L, self.chunk_size, self.norm_service = L, chunk_size, norm_service
        self.device = weights[0].device
        if self.device.type not in ("cpu", "cuda") or (self.device.type == "cuda" and triton is None):
            raise ValueError("MCSR requires CPU or CUDA with Triton")
        master_device = self.device if master_device is None else torch.device(master_device)
        if master_device not in (self.device, torch.device("cpu")):
            raise ValueError("master_device must be CPU or the weight device")
        norm_ptrs = {w.data_ptr() for w in norm_weights}
        self._entries = []
        self._by_ptr = {}
        stores = set()
        for w in weights:
            if not isinstance(w, torch.Tensor):
                raise ValueError("weight must be a tensor")
            _finite_view(w, (torch.bfloat16,), w.shape, (self.device,))
            if not w.numel():
                continue
            ptr = w.data_ptr()
            if ptr in self._by_ptr:
                if w.shape != self._by_ptr[ptr].weight.shape:
                    raise ValueError("tied weights must have identical shapes")
                continue
            if _ident(w) in stores:
                raise ValueError("overlapping weight storage")
            stores.add(_ident(w))
            is_norm = ptr in norm_ptrs
            if is_norm and w.ndim != 1:
                raise ValueError("RMSNorm bypass requires one-dimensional weights")
            mode = "norm" if is_norm else {1: "s", 8: "int4", 128: "int8", MASTER: "master"}[L]
            entry = _Entry(w, mode)
            if mode == "int4":
                entry.residual = torch.zeros((w.numel() + 1) // 2, dtype=torch.uint8, device=w.device)
            if mode == "int8":
                entry.residual = torch.zeros(w.numel(), dtype=torch.int8, device=w.device)
            if mode in ("master", "norm"):
                target = master_device if mode == "master" else self.device
                entry.master = torch.empty(w.shape, dtype=torch.float32, device=target,
                                           pin_memory=w.is_cuda and target.type == "cpu")
                # Copy compact BF16 first when offloading to avoid a full GPU FP32 transient.
                master_input = w.detach().to(target)
                entry.master.view(torch.int32).copy_(master_input.view(torch.int16).int() << 16)
            self._entries.append(entry)
            self._by_ptr[ptr] = entry
        if norm_ptrs - self._by_ptr.keys() or not self._entries:
            raise ValueError("norm weights must belong to the updated weight collection")
        self._validate([(e, None) for e in self._entries], 1.0)
        self._stream_event = torch.cuda.Event() if self.device.type == "cuda" else None
        self._record_stream()

    def _record_stream(self):
        if self._stream_event is not None:
            self._stream_event.record(torch.cuda.current_stream(self.device))

    def _wait_stream(self):
        if self._stream_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._stream_event)

    @property
    def weights(self):
        return tuple(entry.weight for entry in self._entries)

    @property
    def state_bytes(self):
        return sum(_state_size(t) for e in self._entries for t in (e.weight, e.residual, e.master)
                   if t is not None)

    def state_bytes_on(self, device):
        device = torch.device(device)
        return sum(_state_size(t) for e in self._entries for t in (e.weight, e.residual, e.master)
                   if t is not None and t.device == device)

    def norm_master(self, weight):
        entry = self._by_ptr[weight.data_ptr()]
        if entry.mode != "norm":
            raise ValueError("weight is outside the RMSNorm bypass")
        self._wait_stream()
        return entry.master

    def _tiles(self, e, g, chunk_size=None):
        limit = self.chunk_size if chunk_size is None else chunk_size
        for start in range(0, e.weight.numel(), limit):
            stop = min(start + limit, e.weight.numel())
            w = e.weight.view(-1)[start:stop]
            u = None if g is None else g.view(-1)[start:stop]
            r = e.residual
            if e.mode == "int4":
                # For odd chunks the boundary byte is owned by both *coordinates*
                # but writes are ordered in one stream and preserve the other nibble.
                r = r[start // 2:(stop + 1) // 2]
            elif r is not None:
                r = r[start:stop]
            m = None if e.master is None else e.master.view(-1)[start:stop]
            yield start, w, u, r, m

    @staticmethod
    def _offset(e, r, start, n):
        if e.mode == "int8":
            return r.int()
        return _unpack(r, n + (start & 1))[start & 1:]

    @staticmethod
    def _write_offset(e, r, start, offset):
        if e.mode == "int8":
            r.copy_(offset)
            return
        if start & 1:
            r[:1].bitwise_and_(15).bitwise_or_((offset[:1] << 4).to(torch.uint8))
            r, offset = r[1:], offset[1:]
        if offset.numel() % 2:
            r[-1:].bitwise_and_(240).bitwise_or_((offset[-1:] & 15).to(torch.uint8))
            r, offset = r[:-1], offset[:-1]
        _pack_into(r, offset)

    @staticmethod
    def _displacement(g, alpha):
        if g.dtype == torch.float32 and alpha == 1.0:
            return g
        if g.dtype == torch.bfloat16:
            d = (g.view(torch.int16).int() << 16).view(torch.float32)
        else:
            d = g.clone()
        return d if alpha == 1.0 else d.mul_(alpha)

    @torch.no_grad()
    def _validate(self, pending, alpha):
        bad = torch.zeros((), dtype=torch.bool, device=self.device)
        for e, g in pending:
            tile_size = self.chunk_size
            for start, w, u, r, master in self._tiles(e, g, tile_size):
                word = w.view(torch.int16).int()
                mag = word & 0x7FFF
                bad.logical_or_((mag > _BMAX).any())
                if u is not None:
                    d = self._displacement(u, alpha)
                    bad.logical_or_((~torch.isfinite(u)).any())
                    bad.logical_or_((~torch.isfinite(d)).any())
                if r is not None:
                    s = self._offset(e, r, start, w.numel())
                    rank = _rank(word)
                    bad.logical_or_(((s.abs() >= self.L) |
                                     ((rank == _BMAX) & (s > 0)) |
                                     ((rank == -_BMAX) & (s < 0))).any())
                if master is not None:
                    bits = master.to(self.device, non_blocking=True).view(torch.int32)
                    a, b, _ = _mass(bits, 0)
                    rank = _rank(word)
                    bad.logical_or_(((bits & 0x7FFFFFFF) > (_BMAX << 16)).any())
                    bad.logical_or_(((rank != a) & (rank != b)).any())
            if e.mode == "int4" and e.weight.numel() % 2:
                bad.logical_or_((e.residual[-1] & 240) != 0)
        if bool(bad):
            raise ValueError("nonfinite input, out-of-range master, or invalid residual/service state")

    @_serialized
    @torch.no_grad()
    def update_(self, gradients, *, seed, step, alpha=1.0, legacy_chunk=1 << 20):
        """Apply FP32 displacement alpha * gradient, with one preflight and one stats readback.

        Pass unique weights' gradients in state order. Full-step preflight means
        a bad input in a later tensor cannot leave an earlier tensor modified.
        Runtime/device failures require restoration of the last committed checkpoint.
        """
        gradients = tuple(gradients)
        if len(gradients) != len(self._entries) or not math.isfinite(alpha) or legacy_chunk < 1:
            raise ValueError("expected one gradient per unique weight and finite scaling")
        stores = {_ident(t) for e in self._entries for t in (e.weight, e.residual, e.master) if t is not None}
        pending = []
        for e, g in zip(self._entries, gradients):
            if g is None:
                continue
            _finite_view(g, (torch.bfloat16, torch.float32), e.weight.shape, (self.device,))
            if _ident(g) in stores:
                raise ValueError("a gradient aliases mutable state")
            pending.append((e, g))
        self._wait_stream()
        self._validate(pending, alpha)
        # The exponent bins sum the swallowed FP32 magnitudes exactly in
        # units of 2**-149, so the diagnostic also survives a change of tiling.
        counts = torch.zeros(len(STAT_KEYS) + _SUM_EXPONENTS * _SUM_STRIPES,
                             dtype=torch.int64, device=self.device)
        for j, (e, g) in enumerate(zip(self._entries, gradients)):
            if g is None:
                continue
            keys = [_key(seed, step, j, stage) for stage in ("fine", "stay", "move")]
            legacy_gen = None
            if e.mode == "s":
                legacy_gen = torch.Generator(device=self.device).manual_seed(seed * 1000003 + step * 1009 + j)
            size = legacy_chunk if e.mode == "s" else self.chunk_size
            for start, w, u, r, master in self._tiles(e, g, size):
                self._update_tile(e, w, u, r, master, start, alpha, keys, counts, legacy_gen)
        self._record_stream()
        values = counts.tolist()
        result = dict(zip(STAT_KEYS, values))
        bins = values[len(STAT_KEYS):]
        units = sum(sum(bins[i:i + _SUM_STRIPES]) << (i // _SUM_STRIPES)
                    for i in range(0, len(bins), _SUM_STRIPES))
        result["swallowed_abs_sum"] = math.ldexp(float(units), -149)
        result["state_bytes"] = self.state_bytes
        result["changed_bf16_bytes"] = 2 * result["bit_changed_count"]
        result["master_transfer_bytes"] = sum(
            3 * _state_size(e.master) for e, _ in pending
            if e.master is not None and e.master.device != self.device)
        result["step_changed_frac"] = result["bit_changed_count"] / sum(w.numel() for w in self.weights)
        return result

    def _update_tile(self, e, w, u, r, master, start, alpha, keys, counts, legacy_gen):
        word = w.view(torch.int16)
        old_rank = _rank(word.int())
        d = self._displacement(u, alpha)
        if master is not None:
            x = master.to(w.device, non_blocking=True)
        elif r is not None:
            x = _decode(word.int(), self._offset(e, r, start, w.numel()), e.shift)
        else:
            x = (word.int() << 16).view(torch.float32)
        z = x + d
        bits = z.view(torch.int32)
        d_bits = d.view(torch.int32)
        x_bits = x.view(torch.int32)
        zmag = bits & 0x7FFFFFFF
        d_nonzero = (d_bits & 0x7FFFFFFF) != 0
        same = (bits == x_bits) | ((zmag == 0) & ((x_bits & 0x7FFFFFFF) == 0))
        swallowed = same & d_nonzero
        saturated = zmag > (_BMAX << 16)
        counts[2].add_(saturated.sum())
        counts[3].add_(swallowed.sum())
        exp = (d_bits >> 23) & 255
        bin_index = (exp - 1).clamp_min_(0).mul_(_SUM_STRIPES).long()
        lanes = torch.arange(d.numel(), device=d.device, dtype=torch.int64) & (_SUM_STRIPES - 1)
        sig = ((d_bits & 0x7FFFFF) + ((exp != 0).int() << 23)).masked_fill_(~swallowed, 0)
        counts[len(STAT_KEYS):].scatter_add_(0, bin_index.add_(lanes), sig.long())
        del exp, bin_index, lanes, sig
        if e.mode == "s":
            # Match v2.1 S within its finite domain, including its RNG and
            # tile size. New arms use the same finite saturation contract.
            try:
                from .bf16_sr import sr_into
            except ImportError:
                from bf16_sr import sr_into
            z.clamp_(-torch.finfo(torch.bfloat16).max, torch.finfo(torch.bfloat16).max)
            z.masked_fill_((d == 0) & (word == -32768), -0.0)
            dst = torch.empty_like(w)
            sr_into(z, dst, legacy_gen, z.numel())
            new_word = dst.view(torch.int16)
        else:
            bounded = (bits & -0x80000000) | zmag.clamp_max_(_BMAX << 16)
            bounded = torch.where(d_nonzero & ~same, bounded, x_bits)
            del z, bits, zmag, same, swallowed, saturated
            if r is not None:
                bounded = _fine_bits(bounded, e.shift, start, keys[0])
            c, e_rank, q, old_here, keep, rem_near, rem_away = _coupling_masses(
                old_rank, x_bits, bounded, e.shift)
            leave = _uniform(old_here, start, keys[1], bounds=old_here) >= keep
            remaining = rem_near.add_(rem_away).clamp_min_(1)
            moving_away = _uniform(remaining, start, keys[2], bounds=remaining) < rem_away
            new_rank = torch.where(leave, torch.where(moving_away, e_rank, c), old_rank)
            new_word = torch.where(leave, _code(new_rank, bounded < 0).short(), word)
            if r is not None:
                away = (new_rank == e_rank) & (q != 0)
                signed = (q - away.int() * self.L) * (1 - 2 * (bounded < 0).int())
                self._write_offset(e, r, start, signed)
            else:
                master.copy_(bounded.view(torch.float32), non_blocking=True)
        nr = _rank(new_word.int())
        counts[0].add_((new_word != word).sum())
        reverse = ((nr < old_rank) & (d_bits >= 0)) | ((nr > old_rank) & (d_bits < 0))
        counts[1].add_((reverse & d_nonzero).sum())
        word.copy_(new_word)

    @_serialized
    @torch.no_grad()
    def export_master(self, param_idx, *, mode="rtne", seed=0, step=0, device=None):
        """Export one fixed BF16 copy without mutating the master/service pair."""
        e = self._entries[param_idx]
        if e.master is None or mode not in ("rtne", "sr"):
            raise ValueError("master export requires a master and mode rtne or sr")
        self._wait_stream()
        dest = e.master.device if device is None else torch.device(device)
        out = torch.empty(e.weight.shape, device=dest, dtype=torch.bfloat16)
        key = _key(seed, step, param_idx, "export-sr")
        a, b = e.master.view(-1), out.view(-1)
        for i in range(0, a.numel(), self.chunk_size):
            x = a[i:i + self.chunk_size]
            bits = x.view(torch.int32)
            hi, lo = bits >> 16, bits & 65535
            if mode == "rtne":
                carry = (lo > 32768) | ((lo == 32768) & ((hi & 1) != 0))
            else:
                carry = _uniform(hi, i, key, fixed=65536) < lo
            b[i:i + x.numel()].view(torch.int16).copy_((hi + carry.int()).to(dest))
        self._record_stream()
        return out

    @_serialized
    @torch.no_grad()
    def checkpoint(self):
        """CPU copies in physical parameter order, including native BF16 service."""
        self._wait_stream()
        return [{"weight": e.weight.detach().to("cpu", copy=True),
                 "residual": None if e.residual is None else e.residual.to("cpu", copy=True),
                 "master": None if e.master is None else e.master.to("cpu", copy=True)}
                for e in self._entries]

    @_serialized
    @torch.no_grad()
    def restore_(self, entries):
        """Validate a CPU checkpoint before replacing any live state."""
        self._wait_stream()
        if len(entries) != len(self._entries):
            raise ValueError("checkpoint parameter count mismatch")
        staged = []
        for e, src in zip(self._entries, entries):
            ts = []
            for name in ("weight", "residual", "master"):
                dst = getattr(e, name)
                x = src[name]
                if dst is None:
                    if x is not None:
                        raise ValueError("unexpected checkpoint state")
                    ts.append(None)
                else:
                    _finite_view(x, (dst.dtype,), dst.shape, (torch.device("cpu"),))
                    ts.append(x)
            staged.append(_Entry(ts[0], e.mode, ts[1], ts[2]))
        # Validate the saved CPU tensors directly without making a second copy
        # of full-model state on the accelerator.
        validator = object.__new__(MCSR)
        validator.L, validator.chunk_size, validator.device = self.L, self.chunk_size, torch.device("cpu")
        validator._validate([(e, None) for e in staged], 1.0)
        for e, src in zip(self._entries, staged):
            for name in ("weight", "residual", "master"):
                dst = getattr(e, name)
                if dst is not None:
                    dst.copy_(getattr(src, name))
        self._record_stream()


class _FP32NormWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, master):
        return master

    @staticmethod
    def backward(ctx, grad):
        return grad.bfloat16(), None


def rmsnorm_weights(model):
    """Identify 1-D RMSNorm scales, including Qwen2's final model norm."""
    return [module.weight for module in model.modules()
            if type(module).__name__.lower().endswith("rmsnorm")
            and isinstance(getattr(module, "weight", None), torch.nn.Parameter)
            and module.weight.requires_grad and module.weight.ndim == 1]


def install_fp32_norm_service(model, state):
    """Use FP32 gamma in Qwen2-style RMSNorm while retaining BF16 grad sinks."""
    if state.norm_service != "fp32":
        return
    weights = {w.data_ptr() for w in rmsnorm_weights(model)}
    modules = [module for module in model.modules()
               if isinstance(getattr(module, "weight", None), torch.nn.Parameter)
               and module.weight.data_ptr() in weights]
    for module in modules:
        if (type(module).__name__ not in ("Qwen2RMSNorm", "LlamaRMSNorm", "MistralRMSNorm")
                or not hasattr(module, "variance_epsilon")):
            raise ValueError(f"FP32 service is unsupported for {type(module).__name__}")
    for module in modules:
        def forward(hidden_states, norm=module):
            x = hidden_states.float()
            variance = x.square().mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + norm.variance_epsilon)
            master = state.norm_master(norm.weight)
            gamma = _FP32NormWeight.apply(norm.weight, master)
            return (x.to(hidden_states.dtype).float() * gamma).to(hidden_states.dtype)
        module.forward = forward
