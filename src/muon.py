import torch
import torch.distributed as dist


@torch.compile(dynamic=False, fullgraph=True)
def ns5(x):
    """Roughly orthogonalize each matrix of the batch x (n, rows, cols), rows <= cols, in bf16.

    Keller Jordan's quintic Newton-Schulz iteration: https://kellerjordan.github.io/posts/muon/
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = x.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)  # spectral norm at most 1
    for _ in range(5):
        A = x @ x.mT
        x = a * x + (b * A + c * A @ A) @ x
    return x


def lr_scale(fan_in, fan_out):
    """Keller's sqrt(max(1, out/in)), so wide-output matrices like c_fc take bigger steps."""
    return max(1.0, fan_out / fan_in) ** 0.5


class Muon(torch.optim.Optimizer):
    """Muon for the 2D weights in GPT-2's blocks, with the Newton-Schulz work split over the ranks.

    blocks holds (attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj) weights per layer, stored by HF's
    Conv1D as (in, out). Q, K and V are orthogonalized as separate matrices. Rank r keeps momentum
    for and orthogonalizes only its own contiguous share of the layers, then the bf16 updates are
    all-gathered, so every rank applies the same update to every weight and they stay in sync.
    Grads must already be averaged over the ranks (DDP does it).
    """

    def __init__(self, blocks, lr, weight_decay, momentum):
        super().__init__([p for block in blocks for p in block],
                         dict(lr=lr, weight_decay=weight_decay, momentum=momentum))
        self.blocks = blocks
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        assert len(blocks) % self.world == 0, "the layers must split evenly over the ranks"
        per_rank = len(blocks) // self.world
        self.mine = [p for block in blocks[self.rank * per_rank:(self.rank + 1) * per_rank] for p in block]
        for p in self.mine:
            self.state[p]["momentum"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None, "Muon does not take a closure"  # the parameter matches Optimizer.step
        group = self.param_groups[0]
        lr, weight_decay, momentum = group["lr"], group["weight_decay"], group["momentum"]

        # Nesterov momentum on this rank's own weights.
        grads = [p.grad for p in self.mine]
        bufs = [self.state[p]["momentum"] for p in self.mine]
        torch._foreach_lerp_(bufs, grads, 1 - momentum)
        updates = torch._foreach_lerp(grads, bufs, momentum)

        # Two batches of same-shaped matrices: Q, K, V and attn.c_proj as (d, d); c_fc and the
        # transposed mlp.c_proj as (d, 4d). Orthogonalizing the transpose gives the transposed result.
        square, wide = [], []
        for c_attn, attn_proj, c_fc, mlp_proj in zip(*[iter(updates)] * 4):
            d = c_attn.shape[0]
            square += [*c_attn.view(d, 3, d).unbind(1), attn_proj]
            wide += [c_fc, mlp_proj.T]
        gathered = []
        for mats in (square, wide):  # the square gather runs while the wide batch is orthogonalized
            local = ns5(torch.stack(mats))
            out = local.new_empty((self.world * len(mats), *local.shape[1:]))
            work = dist.all_gather_into_tensor(out, local, async_op=True)
            assert work is not None  # async_op=True always returns a handle
            gathered.append((work, out))
        for work, _ in gathered:
            work.wait()
        square, wide = (out for _, out in gathered)

        # Decoupled weight decay, then the scaled update, the same on every rank.
        torch._foreach_mul_(self.param_groups[0]["params"], 1 - lr * weight_decay)
        for i, (c_attn, attn_proj, c_fc, mlp_proj) in enumerate(self.blocks):
            d = c_attn.shape[0]
            c_attn.view(d, 3, d).add_(square[4 * i:4 * i + 3].transpose(0, 1), alpha=-lr * lr_scale(d, d))
            attn_proj.add_(square[4 * i + 3], alpha=-lr * lr_scale(*attn_proj.shape))
            c_fc.add_(wide[2 * i], alpha=-lr * lr_scale(*c_fc.shape))
            mlp_proj.add_(wide[2 * i + 1].T, alpha=-lr * lr_scale(*mlp_proj.shape))
