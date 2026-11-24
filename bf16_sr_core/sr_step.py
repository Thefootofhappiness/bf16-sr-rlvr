# sr_step.py — Exact-global-clip Masterless BF16-SR GRPO step with vocab-tiled FP32 head & optional CPU grad offload
import math
import torch
from bf16_sr import sr_into

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def blocks(x: torch.Tensor, size: int):
    a = x.view(-1)
    for i in range(0, a.numel(), size):
        yield i, a[i : i + size]


def grad_sqnorm(g: torch.Tensor, chunk: int) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=g.device)
    for _, part in blocks(g, chunk):
        v = part.double()
        total.add_(torch.dot(v, v))
    return total


@torch.no_grad()
def write_param(p: torch.Tensor, g: torch.Tensor, alpha: float, gen: torch.Generator, chunk: int, emit=None, stats=None) -> None:
    w = p.view(-1)
    for i, gb in blocks(g, chunk):
        old = w[i : i + gb.numel()]
        d = gb.to(device=p.device, dtype=torch.float32, copy=True).mul_(alpha)
        y = old.float().add_(d)
        # Preserve -0.0 bit pattern when d == 0 (consistent with original full-SR gate).
        y.masked_fill_((d == 0) & (old.view(torch.int16) == -32768), -0.0)
        if stats is not None:
            stats[1].add_(torch.sum(d.square(), dtype=torch.float64))
        del d
        dst = old if emit is None and stats is None else torch.empty_like(old)
        sr_into(y, dst, gen, chunk)
        if emit is not None:
            emit(i, old, dst)
        if stats is not None:
            stats[0].add_(torch.count_nonzero(dst.view(torch.int16) != old.view(torch.int16)))
        if dst is not old:
            old.copy_(dst)


class StepGrads:
    def __init__(self, model: torch.nn.Module, host: bool = False):
        self.ps = [p for p in model.parameters() if p.requires_grad]
        if not self.ps or any(
            p.dtype != torch.bfloat16 or not p.is_contiguous() or p.device != self.ps[0].device
            for p in self.ps
        ):
            raise ValueError("one device, contiguous trainable BF16 parameters required")
        ptrs = [p.untyped_storage().data_ptr() for p in self.ps]
        if len(set(ptrs)) != len(ptrs):
            raise ValueError("shared storage must use the same Parameter object")
        self.host = host
        self.buf = {}
        self.hooks = []

    def __enter__(self):
        for p in self.ps:
            p.grad = None
            if self.host:
                self.hooks.append(p.register_post_accumulate_grad_hook(self.capture))
        return self

    def __exit__(self, typ, value, tb):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @torch.no_grad()
    def capture(self, p: torch.Tensor):
        g = p.grad.detach().cpu()
        if p in self.buf:
            self.buf[p].add_(g)
        else:
            self.buf[p] = g
        p.grad = None

    @torch.no_grad()
    def add_head(self, p: torch.Tensor, first_row: int, dw: torch.Tensor):
        if self.host:
            if p not in self.buf:
                self.buf[p] = torch.zeros_like(p, device="cpu")
            dest = self.buf[p]
        else:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dest = p.grad
        a = dest[first_row : first_row + dw.shape[0]]
        a.copy_(a.float().add_(dw.to(a.device)))

    @torch.no_grad()
    def finish(self, eta: float, seed: int, step: int, chunk: int, emit=None):
        gs = [self.buf.get(p) if self.host else p.grad for p in self.ps]
        squares = [grad_sqnorm(g, chunk) for g in gs if g is not None]
        if not squares:
            return 0.0, 0.0, 0.0, False
        norm = torch.stack(squares).sum().sqrt().item()
        if not math.isfinite(norm):
            raise FloatingPointError("nonfinite global norm before writeback")
        if norm == 0.0:
            for p in self.ps:
                p.grad = None
            self.buf.clear()
            return norm, 0.0, 0.0, False

        alpha = -eta * min(1.0, 1.0 / norm)
        dev = self.ps[0].device
        stats = (
            torch.zeros((), device=dev, dtype=torch.int64),
            torch.zeros((), device=dev, dtype=torch.float64),
        )
        for j, (p, g) in enumerate(zip(self.ps, gs)):
            if g is not None:
                gen = torch.Generator(device=p.device).manual_seed(seed * 1000003 + step * 1009 + j)
                callback = None
                if emit is not None:
                    def callback(offset, old, new, param_id=j):
                        emit(param_id, offset, old, new)
                write_param(p, g, alpha, gen, chunk, callback, stats)
            p.grad = None
            gs[j] = None
        self.buf.clear()
        energy = stats[1].item()
        changed = stats[0].item() / sum(p.numel() for p in self.ps)
        return norm, changed, energy / max(energy, 1e-30), True


@torch.no_grad()
def head_tiles(h: torch.Tensor, w: torch.Tensor, ids: torch.Tensor, adv_over_n: float, sink: StepGrads, tied: bool, rows: int) -> None:
    x = h.detach().float()
    log_z = torch.full_like(x[:, 0], -math.inf)

    def tiles():
        for j in range(0, w.shape[0], rows):
            yield j, w[j : j + rows].detach().float()

    for _, a in tiles():
        log_z = torch.logaddexp(log_z, torch.logsumexp(x @ a.T, dim=1))

    def derivative(j: int, a: torch.Tensor) -> torch.Tensor:
        dz = (x @ a.T).sub_(log_z[:, None]).exp_()
        col = ids - j
        valid = (col >= 0) & (col < a.shape[0])
        dz.scatter_add_(1, col.clamp(0, a.shape[0] - 1)[:, None], -valid[:, None].to(dz.dtype))
        return dz.mul_(adv_over_n)

    dh = torch.zeros_like(x)
    for j, a in tiles():
        dz = derivative(j, a)
        dh.addmm_(dz, a)
        if not tied:
            sink.add_head(w, j, dz.T @ x)

    h.backward(dh.to(h.dtype))
    if tied:
        for j, a in tiles():
            sink.add_head(w, j, derivative(j, a).T @ x)


def full_sr_step(model, records, eta: float, seed: int, step: int, host: bool = False, rows: int = 2048, chunk: int = 1 << 20, emit=None):
    if rows < 1 or chunk < 1:
        raise ValueError("tile sizes must be positive")
    model.train()
    model.gradient_checkpointing_enable()
    w = model.get_output_embeddings().weight
    tied = w is model.get_input_embeddings().weight
    nt = sum(len(c) for rec in records for c in rec["comp_ids"])
    with StepGrads(model, host) as sink:
        for rec in records:
            prompt = rec["prompt_ids"]
            for c, adv in zip(rec["comp_ids"], rec["advs"]):
                if not c or not adv:
                    continue
                ids = torch.tensor(prompt + c, device=w.device)
                hidden = model.model(input_ids=ids[None], use_cache=False).last_hidden_state
                h = hidden[0, len(prompt) - 1 : -1]
                head_tiles(h, w, ids[len(prompt) :], adv / nt, sink, tied, rows)
        return sink.finish(eta, seed, step, chunk, emit)
