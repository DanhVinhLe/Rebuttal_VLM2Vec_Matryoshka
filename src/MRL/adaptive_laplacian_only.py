from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaptiveLaplacianOnlyStage1Loss(nn.Module):
    """Adaptive MRL stage-1 Laplacian + base contrastive-per-dim loss."""

    def __init__(self, args):
        super().__init__()
        self.nested_dims = sorted(set(getattr(args, "nested_dims", None) or [64, 128, 256, 512, 768, 1024]))
        self.average_loss = bool(getattr(args, "average_loss", True))
        self.temperature = float(getattr(args, "temperature", 0.02))

        self.spectrum_kl_eps = float(getattr(args, "spectrum_kl_eps", 1e-8))
        self.spectrum_kl_weight = float(getattr(args, "spectrum_kl_weight", 1.0))
        self.laplacian_tau = float(getattr(args, "laplacian_tau", 0.07))
        self.laplacian_k_eig = int(getattr(args, "laplacian_k_eig", 10))
        laplacian_top_k = int(getattr(args, "laplacian_top_k", -1))
        self.laplacian_top_k: Optional[int] = laplacian_top_k if laplacian_top_k > 0 else None
        self.laplacian_pair_weights = self._parse_pair_weight_map(getattr(args, "laplacian_pair_weights", ""))

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

    def _dist_gather_tensor(self, t: Tensor) -> Tensor:
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def _resolve_dims(self, full_dim: int) -> List[int]:
        valid_dims = [d for d in self.nested_dims if d <= full_dim]
        if full_dim not in valid_dims:
            valid_dims.append(full_dim)
        return sorted(set(valid_dims))

    def _build_contrastive_target(self, q: Tensor, p: Tensor) -> Tensor:
        target = torch.arange(q.size(0), device=q.device, dtype=torch.long)
        target_per_qry = p.size(0) // q.size(0)
        return target * target_per_qry

    def _parse_pair_weight_map(self, spec) -> Dict[tuple[int, int], float]:
        if not spec:
            return {}
        out = {}
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            if "->" in item and ":" in item:
                pair_spec, weight_spec = item.rsplit(":", 1)
                src_str, dst_str = pair_spec.split("->", 1)
            else:
                src_str, dst_str, weight_spec = item.split(":")
            out[(int(src_str.strip()), int(dst_str.strip()))] = float(weight_spec.strip())
        return out

    def compute_laplacian(self, z: Tensor, tau: float, top_k: Optional[int] = None) -> Tensor:
        z = F.normalize(z, p=2, dim=-1, eps=self.spectrum_kl_eps)
        sim = z @ z.t()
        adj = torch.exp(sim / max(tau, self.spectrum_kl_eps))
        if top_k is not None:
            top_k = max(1, min(int(top_k), adj.size(-1)))
            keep_idx = torch.topk(adj, k=top_k, dim=-1, largest=True, sorted=False).indices
            keep_mask = torch.zeros_like(adj, dtype=torch.bool)
            keep_mask.scatter_(dim=-1, index=keep_idx, value=True)
            keep_mask = keep_mask | keep_mask.t()
            adj = adj * keep_mask.to(adj.dtype)
        degree = adj.sum(dim=-1)
        d_inv = torch.rsqrt(degree + self.spectrum_kl_eps)
        eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        return eye - (d_inv[:, None] * adj * d_inv[None, :])

    def compute_spectrum(self, laplacian: Tensor, k_eig: int) -> Tensor:
        eigvals = torch.linalg.eigh(
            laplacian.float() if laplacian.dtype in (torch.bfloat16, torch.float16) else laplacian
        ).eigenvalues
        eigvals = eigvals.clamp_min(self.spectrum_kl_eps)
        k_eff = max(1, min(int(k_eig), eigvals.numel()))
        low = eigvals[:k_eff]
        return (low / (low.sum() + self.spectrum_kl_eps)).clamp_min(self.spectrum_kl_eps)

    def spectral_loss(self, z_small: Tensor, z_large: Tensor) -> Tensor:
        p_small = self.compute_spectrum(
            self.compute_laplacian(z_small, self.laplacian_tau, self.laplacian_top_k),
            self.laplacian_k_eig,
        )
        p_large = self.compute_spectrum(
            self.compute_laplacian(z_large, self.laplacian_tau, self.laplacian_top_k),
            self.laplacian_k_eig,
        ).detach()
        p_small = p_small / (p_small.sum() + self.spectrum_kl_eps)
        p_large = p_large / (p_large.sum() + self.spectrum_kl_eps)
        return torch.sum(
            p_small * torch.log((p_small + self.spectrum_kl_eps) / (p_large + self.spectrum_kl_eps))
        ) + torch.sum(
            p_large * torch.log((p_large + self.spectrum_kl_eps) / (p_small + self.spectrum_kl_eps))
        )

    def forward(self, model_trainer, input_data):
        model = model_trainer.model
        qry_full = model.encode_input(input_data["qry"])[0]
        pos_full = model.encode_input(input_data["pos"])[0]

        if self.world_size > 1:
            qry_full = self._dist_gather_tensor(qry_full)
            pos_full = self._dist_gather_tensor(pos_full)

        dims = self._resolve_dims(qry_full.size(-1))
        target = self._build_contrastive_target(qry_full, pos_full)

        # 1) Base contrastive loss per dim (same core behavior as base_mrl.py)
        contrastive_terms = []
        dim_metrics: Dict[str, Tensor] = {}
        for dim in dims:
            q = F.normalize(qry_full[:, :dim], p=2, dim=-1)
            p = F.normalize(pos_full[:, :dim], p=2, dim=-1)
            scores = model.compute_similarity(q, p).view(q.size(0), -1)
            loss_dim = F.cross_entropy(scores / self.temperature, target)
            contrastive_terms.append(loss_dim)
            dim_metrics[f"contrastive_loss_dim_{dim}"] = loss_dim.detach()

        if contrastive_terms:
            contrastive_loss = torch.stack(contrastive_terms).mean() if self.average_loss else torch.stack(contrastive_terms).sum()
        else:
            contrastive_loss = torch.zeros((), device=qry_full.device, dtype=qry_full.dtype)

        # 2) Extra Laplacian consistency term across adjacent dims
        if len(dims) >= 2:
            lap_terms, weights = [], []
            for small_dim, large_dim in zip(dims[:-1], dims[1:]):
                small_rep = torch.cat([qry_full[:, :small_dim], pos_full[:, :small_dim]], dim=0)
                large_rep = torch.cat([qry_full[:, :large_dim], pos_full[:, :large_dim]], dim=0)
                lap_terms.append(self.spectral_loss(small_rep, large_rep))
                weights.append(self.laplacian_pair_weights.get((large_dim, small_dim), 1.0))
            loss_stack = torch.stack(lap_terms)
            w = torch.tensor(weights, device=loss_stack.device, dtype=loss_stack.dtype)
            spectrum_kl_loss = (loss_stack * w).sum() / w.sum().clamp_min(self.spectrum_kl_eps)
        else:
            spectrum_kl_loss = torch.zeros((), device=qry_full.device, dtype=qry_full.dtype)

        total_loss = contrastive_loss + self.spectrum_kl_weight * spectrum_kl_loss

        result = {
            "loss": total_loss,
            "total_loss": total_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "align_loss": contrastive_loss.detach(),
            "orthogonal_loss": torch.zeros_like(total_loss).detach(),
            "spectrum_kl_loss": spectrum_kl_loss.detach(),
            "spectrum_kl_weight": torch.tensor(self.spectrum_kl_weight, device=total_loss.device),
        }
        result.update(dim_metrics)
        return result
