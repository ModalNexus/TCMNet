"""TCMCoarse refinement for coarse matching.

This module inserts a *sparse* neighborhood-consensus refinement between the
coarse similarity matrix and the dual-softmax in
:class:`~models.pair_matcher.CoarseMatching`.

Motivation
----------
Cross-modal (modalities 1/2/3) appearance similarity is unreliable: the dual-softmax
confidence of a *true* coarse correspondence is often low (recall loss), while
repetitive structures create isolated high-similarity false peaks (precision
loss). Both are hard for the fine stage to fix, because fine matching only
polishes a match inside a small window centred on the coarse cell -- it cannot
relocate a wrong coarse cell.

We therefore add a learnable *local geometric consensus* prior directly on the
coarse similarity: for every candidate pair ``(i, j)`` we look at the small
``w x w`` neighbourhood around ``i`` in image0 and around ``j`` in image1,
gather the local ``[w^2, w^2]`` correlation block, and run a tiny 4D
convolution over it. A true match exhibits a coherent "diagonal" structure in
that block (neighbours of ``i`` match neighbours of ``j`` with a consistent
offset); an isolated false peak does not. The 4D conv learns to score this.

The re-scoring is applied as a **residual** ``sim' = sim + alpha * delta`` and
the final conv layer is **zero-initialised**, so at training step 0 the module
is exactly the identity and reproduces the pretrained pairwise matcher output
byte-for-byte. Only the candidate entries are ever modified; every other entry
of the similarity matrix is untouched.

Design notes
------------
* Candidate selection is performed under ``torch.no_grad`` (it is argmax/topk,
  non-differentiable); only the *scored* delta values carry gradient.
* During training the ground-truth coarse pairs are injected into the candidate
  set so the module receives gradient on true matches even when the pretrained
  backbone ranks them below the top-k -- this is what drives the recall gain.
* Memory stays bounded: instead of the dense ``[N, h0, w0, h1, w1]`` 4D tensor
  (prohibitive at 64x64), we only materialise ``[P, 1, w, w, w, w]`` for the
  ``P`` selected candidates.
"""

import torch
import torch.nn as nn

# similarity padding fill used by CoarseMatching (masked_fill_ with -INF)
_PAD_FILL = -1e8


class Conv4d(nn.Module):
    """A 4D convolution implemented as a sum of 3D convolutions.

    PyTorch has no native ``Conv4d``. We convolve the last three spatial axes
    with a :class:`~torch.nn.Conv3d` for every offset along the first spatial
    axis, then shift-and-accumulate. This is the standard trick used by
    neighbourhood-consensus networks and is cheap here because every spatial
    axis has size ``w`` (typically 5).

    Input / output shape: ``[B, C_in, D1, D2, D3, D4]`` -> ``[B, C_out, D1, D2,
    D3, D4]`` with 'same' padding.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.k = kernel_size
        self.pad = kernel_size // 2
        self.out_ch = out_ch
        # one Conv3d per offset along the first spatial axis; only the centre
        # convolution carries the bias so the summed output has a single bias.
        self.convs = nn.ModuleList(
            [
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=self.pad,
                    bias=(bias and t == self.pad),
                )
                for t in range(kernel_size)
            ]
        )

    def zero_init(self) -> None:
        """Zero every weight/bias so the layer outputs 0 (identity residual)."""
        for conv in self.convs:
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, D1, D2, D3, D4 = x.shape
        out = x.new_zeros(B, self.out_ch, D1, D2, D3, D4)
        # flatten the D1 axis into the batch so a single Conv3d handles all
        # D1 slices at once, then reshape back.
        x_slices = x.permute(0, 2, 1, 3, 4, 5).reshape(B * D1, C_in, D2, D3, D4)
        for t, conv in enumerate(self.convs):
            offset = t - self.pad  # input slice offset relative to output slice
            y = conv(x_slices).reshape(B, D1, self.out_ch, D2, D3, D4)
            y = y.permute(0, 2, 1, 3, 4, 5)  # [B, C_out, D1, D2, D3, D4]
            if offset == 0:
                out = out + y
            elif offset > 0:
                out[:, :, : D1 - offset] = out[:, :, : D1 - offset] + y[:, :, offset:]
            else:
                o = -offset
                out[:, :, o:] = out[:, :, o:] + y[:, :, : D1 - o]
        return out


class TCMCoarse(nn.Module):
    """Sparse, learnable local-consensus re-scoring of coarse similarity.

    Args:
        config (dict): the lowered ``match_coarse`` config dict. Reads the keys
            ``tcm_coarse_enable``, ``tcm_coarse_window``, ``tcm_coarse_topk``,
            ``tcm_coarse_max_candidates``, ``tcm_coarse_channels``, ``tcm_coarse_kernel``,
            ``tcm_coarse_alpha`` (all optional, so old
            configs without them keep the module disabled).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.enable = bool(config.get("tcm_coarse_enable", False))
        self.w = int(config.get("tcm_coarse_window", 5))
        assert self.w % 2 == 1, "tcm_coarse_window must be odd"
        self.topk = int(config.get("tcm_coarse_topk", 2))
        self.max_cand = int(config.get("tcm_coarse_max_candidates", 4096))
        self.alpha = float(config.get("tcm_coarse_alpha", 1.0))
        ch = int(config.get("tcm_coarse_channels", 8))
        k = int(config.get("tcm_coarse_kernel", 3))
        self.pad_val = -1e4  # sentinel for out-of-image / padded neighbours

        # Only build (parameterised) layers when enabled. This keeps the
        # disabled path byte-identical and keeps ``tcm_coarse.*`` out of the
        # state_dict, so pretrained/pre-NC checkpoints still load with
        # strict=True.
        if self.enable:
            self.conv = nn.Sequential(
                Conv4d(1, ch, k),
                nn.ReLU(inplace=True),
                Conv4d(ch, ch, k),
                nn.ReLU(inplace=True),
                Conv4d(ch, 1, k),
            )
            # zero-init the last conv => delta == 0 at init => exact identity.
            self.conv[-1].zero_init()

    @torch.no_grad()
    def _select_candidates(self, sim, data, mask_c0, mask_c1):
        """Pick candidate (b, i, j) triples to re-score.

        Superset = top-k columns per row + top-k rows per column (softmax is
        row/col-monotonic, so top-k on the raw ``sim`` matches top-k on the
        dual-softmax confidence). Ground-truth pairs are injected during
        training. Candidates whose centre falls in a padded row/col are dropped.
        """
        N, L, S = sim.shape
        device = sim.device

        k_col = min(self.topk, S)
        top_j = sim.topk(k_col, dim=2).indices  # [N, L, k_col]
        b_a = torch.arange(N, device=device)[:, None, None].expand(N, L, k_col)
        i_a = torch.arange(L, device=device)[None, :, None].expand(N, L, k_col)
        cand_a = torch.stack([b_a.reshape(-1), i_a.reshape(-1), top_j.reshape(-1)], dim=1)

        k_row = min(self.topk, L)
        top_i = sim.topk(k_row, dim=1).indices  # [N, k_row, S]
        b_b = torch.arange(N, device=device)[:, None, None].expand(N, k_row, S)
        j_b = torch.arange(S, device=device)[None, None, :].expand(N, k_row, S)
        cand_b = torch.stack([b_b.reshape(-1), top_i.reshape(-1), j_b.reshape(-1)], dim=1)

        cand = torch.cat([cand_a, cand_b], dim=0)

        gt = None
        if self.training and "spv_b_ids" in data:
            gt = torch.stack([data["spv_b_ids"], data["spv_i_ids"], data["spv_j_ids"]], dim=1)
            cand = torch.cat([cand, gt.to(cand.dtype)], dim=0)

        cand = torch.unique(cand, dim=0)
        b, i, j = cand[:, 0], cand[:, 1], cand[:, 2]

        if mask_c0 is not None:
            keep = mask_c0[b, i].bool() & mask_c1[b, j].bool()
            b, i, j = b[keep], i[keep], j[keep]

        if self.max_cand > 0:
            b, i, j = self._cap_per_batch(b, i, j, sim, gt)
        return b, i, j

    @torch.no_grad()
    def _cap_per_batch(self, b, i, j, sim, gt=None):
        """Cap candidates deterministically at inference and preserve training GT.

        The training cap is soft when valid GT alone exceeds it. All remaining
        training slots are randomly sampled; inference keeps highest logits,
        with stable row/column order breaking equal-score ties.
        """
        parts_b, parts_i, parts_j = [], [], []
        for n in range(sim.shape[0]):
            sel = (b == n).nonzero(as_tuple=True)[0]
            if sel.numel() > self.max_cand:
                if self.training:
                    protected = torch.zeros(sel.numel(), dtype=torch.bool, device=b.device)
                    if gt is not None:
                        gt_n = gt[gt[:, 0] == n]
                        protected = torch.isin(
                            i[sel] * sim.shape[2] + j[sel],
                            gt_n[:, 1] * sim.shape[2] + gt_n[:, 2],
                        )
                    kept_gt = sel[protected]
                    others = sel[~protected]
                    remaining = max(0, self.max_cand - kept_gt.numel())
                    order = torch.randperm(others.numel(), device=b.device)[:remaining]
                    sel = torch.cat((kept_gt, others[order]))
                else:
                    order = torch.argsort(sim[b[sel], i[sel], j[sel]], descending=True, stable=True)
                    sel = sel[order[: self.max_cand]]
            parts_b.append(b[sel])
            parts_i.append(i[sel])
            parts_j.append(j[sel])
        return torch.cat(parts_b), torch.cat(parts_i), torch.cat(parts_j)

    def _gather_block(self, sim, b, i, j, hw0, hw1):
        """Gather the local ``[P, 1, w, w, w, w]`` correlation block per candidate."""
        H0, W0 = int(hw0[0]), int(hw0[1])
        H1, W1 = int(hw1[0]), int(hw1[1])
        w, r = self.w, self.w // 2
        device = sim.device

        off = torch.arange(-r, r + 1, device=device)  # [w]
        dy = off.reshape(w, 1).expand(w, w).reshape(-1)  # [w*w]
        dx = off.reshape(1, w).expand(w, w).reshape(-1)  # [w*w]

        y0 = torch.div(i, W0, rounding_mode="trunc")
        x0 = i % W0
        y1 = torch.div(j, W1, rounding_mode="trunc")
        x1 = j % W1

        ny0 = y0[:, None] + dy[None, :]
        nx0 = x0[:, None] + dx[None, :]  # [P, w*w]
        ny1 = y1[:, None] + dy[None, :]
        nx1 = x1[:, None] + dx[None, :]

        valid0 = (ny0 >= 0) & (ny0 < H0) & (nx0 >= 0) & (nx0 < W0)
        valid1 = (ny1 >= 0) & (ny1 < H1) & (nx1 >= 0) & (nx1 < W1)

        idx0 = ny0.clamp(0, H0 - 1) * W0 + nx0.clamp(0, W0 - 1)  # [P, w*w]
        idx1 = ny1.clamp(0, H1 - 1) * W1 + nx1.clamp(0, W1 - 1)  # [P, w*w]

        # advanced indexing broadcasts to [P, w*w, w*w]
        block = sim[b[:, None, None], idx0[:, :, None], idx1[:, None, :]]

        vmask = valid0[:, :, None] & valid1[:, None, :]
        block = torch.where(vmask & (block > _PAD_FILL), block, block.new_full((), self.pad_val))
        return block.reshape(-1, 1, w, w, w, w)

    def forward(self, sim, data, mask_c0=None, mask_c1=None):
        """Return a dense residual ``delta`` (same shape as ``sim``) or ``None``.

        ``None`` means "no change" (module disabled or no candidates); the caller
        then leaves ``sim`` untouched.
        """
        if not self.enable:
            return None

        b, i, j = self._select_candidates(sim, data, mask_c0, mask_c1)
        if b.numel() == 0:
            return None

        block = self._gather_block(sim, b, i, j, data["hw0_c"], data["hw1_c"])
        out = self.conv(block)  # [P, 1, w, w, w, w]; 0 at init
        c = self.w // 2
        delta = out[:, 0, c, c, c, c]  # [P] re-score for each (i, j)

        delta_dense = sim.new_zeros(sim.shape)
        delta_dense.index_put_((b, i, j), self.alpha * delta, accumulate=False)
        return delta_dense
