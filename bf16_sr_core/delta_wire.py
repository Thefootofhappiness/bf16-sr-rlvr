# delta_wire.py — 64-bit tile-offset sparse delta-sync wire protocol (COO / gap / bitmask / dense + optional codec)
import struct
import zlib
import numpy as np
import torch

# base version, new version, parameter ID, element offset (uint64), tile length,
# changed count, representation, codec, wire length, CRC32
HEADER = struct.Struct("!QQIQIIBBII")


def codec_op(data: bytes, codec: int, encode: bool) -> bytes:
    if codec == 0:
        return data
    if codec == 1:
        return zlib.compress(data, 1) if encode else zlib.decompress(data)
    if codec == 2:
        import zstandard as zstd
        obj = zstd.ZstdCompressor(level=1) if encode else zstd.ZstdDecompressor()
        return obj.compress(data) if encode else obj.decompress(data)
    if codec == 3:
        import lz4.frame as lz4
        return lz4.compress(data) if encode else lz4.decompress(data)
    raise ValueError("unknown codec")


def wire_array(x: torch.Tensor, dtype: str) -> bytes:
    return x.detach().cpu().numpy().astype(dtype, copy=False).tobytes()


@torch.no_grad()
def packet(base: int, version: int, pi: int, start: int, old: torch.Tensor, new: torch.Tensor, mode: str = "auto", codec: int = 0) -> bytes:
    a = old.view(torch.int16).view(-1)
    b = new.view(torch.int16).view(-1)
    n = a.numel()
    if not 0 < n < 2**31 or b.shape != a.shape:
        raise ValueError("invalid tile")

    mask = a != b
    idx = mask.nonzero().flatten()
    k = idx.numel()
    if mode == "auto":
        costs = {"coo": 6 * k, "mask": (n + 7) // 8 + 2 * k, "dense": 2 * n}
        mode = min(costs, key=costs.get)
    kind = ("coo", "gap", "mask", "dense").index(mode)

    if mode == "dense":
        data = wire_array(b, "<i2")
    else:
        values = wire_array(b[idx], "<i2")
        if mode in ("coo", "gap"):
            pos = idx.int()
            if mode == "gap" and k > 1:
                pos = torch.cat((pos[:1], pos[1:] - pos[:-1]))
            meta = wire_array(pos, "<i4")
        else:
            bits = torch.nn.functional.pad(mask, (0, (-n) % 8)).view(-1, 8)
            shifts = torch.arange(8, device=b.device, dtype=torch.int32)
            packed = (bits.int() << shifts).sum(1).to(torch.uint8)
            meta = wire_array(packed, "u1")
        data = meta + values

    wire = codec_op(data, codec, True)
    if len(wire) >= len(data):
        wire, codec = data, 0
    hdr = HEADER.pack(base, version, pi, start, n, k, kind, codec, len(wire), zlib.crc32(wire))
    return hdr + wire


def read_exact(sock, size: int) -> bytearray:
    out = bytearray(size)
    view = memoryview(out)
    while view:
        count = sock.recv_into(view)
        if not count:
            raise EOFError("incomplete delta")
        view = view[count:]
    return out


@torch.no_grad()
def receive_tile(sock, dst: torch.Tensor, expected: tuple) -> None:
    base, version, pi, start, n, k, kind, codec, size, crc = HEADER.unpack(read_exact(sock, HEADER.size))
    if (
        (base, version, pi, start, n) != expected
        or k > n
        or kind > 3
        or codec > 3
        or size > 6 * n + (n + 7) // 8
    ):
        raise ValueError("version, layout or size mismatch")
    wire = read_exact(sock, size)
    if zlib.crc32(wire) != crc:
        raise ValueError("delta CRC mismatch")

    data = codec_op(bytes(wire), codec, False)
    w = dst.view(torch.int16).view(-1).narrow(0, start, n)
    meta = (4 * k, 4 * k, (n + 7) // 8, 0)[kind]
    if len(data) != meta + 2 * (n if kind == 3 else k):
        raise ValueError("invalid payload length")
    vals = np.frombuffer(data, "<i2", offset=meta).astype(np.int16, copy=True)
    v = torch.from_numpy(vals).to(w.device)
    if kind == 3:
        w.copy_(v)
        return

    if kind == 2:
        packed = np.frombuffer(data, "u1", count=meta)
        pos = np.flatnonzero(np.unpackbits(packed, bitorder="little")[:n])
    else:
        pos = np.frombuffer(data, "<i4", count=k).astype(np.int64)
        if kind == 1:
            pos = np.cumsum(pos)
    if len(pos) != k or (k and (pos[0] < 0 or pos[-1] >= n or np.any(pos[1:] <= pos[:-1]))):
        raise ValueError("invalid positions")
    w[torch.from_numpy(pos).to(w.device)] = v
