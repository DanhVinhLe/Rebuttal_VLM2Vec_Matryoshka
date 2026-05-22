from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaptiveProjectionOnlyStage1Loss(nn.Module):
    """Adaptive MRL stage-1 projection/alignment loss only (no spectrum/laplacian term)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temperature = getattr(args, "temperature", 0.02)
        nested_dims = getattr(args, "nested_dims", None) or [64, 128, 256, 512, 768, 1024]
        self.nested_dims = sorted(set(nested_dims))
        self.phase = str(getattr(args, "stage1_phase", "all")).upper()
        self.projection_spec = str(getattr(args, "stage1_projection_spec", "")).strip()
        self.align_l1_weight = float(getattr(args, "align_l1_weight", 1.0))
        self.full_dim_l1_weight = float(getattr(args, "full_dim_l1_weight", 0.0))
        self.orthogonal_weight = float(getattr(args, "orthogonal_weight", 0.01))
        self.orthogonal_pair_weights = self._parse_pair_weight_map(getattr(args, "orthogonal_pair_weights", ""))
        self.projection_weights = self._parse_pair_weight_map(getattr(args, "stage1_projection_weights", ""))
        self.dim_align_l1_weights = self._parse_dim_weight_map(getattr(args, "align_l1_weights", ""))

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

    def _build_contrastive_target(self, q: Tensor, p: Tensor) -> Tensor:
        target = torch.arange(q.size(0), device=q.device, dtype=torch.long)
        target_per_qry = p.size(0) // q.size(0)
        return target * target_per_qry

    def _project_to_dim(self, model, x: Tensor, dim: int, src_dim: Optional[int] = None) -> Tensor:
        if src_dim is None:
            src_dim = x.size(-1)
        if src_dim == dim:
            return x[:, :dim]
        if src_dim < dim:
            raise ValueError(f"Cannot project {src_dim} -> {dim}: source dim is smaller.")
        if not hasattr(model, "matryoshka_proj_bank"):
            raise RuntimeError("Model missing `matryoshka_proj_bank`. Attach it before stage1 training.")
        return model.matryoshka_proj_bank.project(x[:, :src_dim], src_dim=src_dim, dst_dim=dim)

    def _cross_alignment_l1(self, model, qry: Tensor, pos: Tensor, target: Tensor, dim: int, bigger_dim: Optional[int] = None):
        if bigger_dim is None:
            bigger_dim = dim
        q_dim = F.normalize(self._project_to_dim(model, qry, dim, src_dim=bigger_dim), p=2, dim=-1)
        p_dim = F.normalize(self._project_to_dim(model, pos, dim, src_dim=bigger_dim), p=2, dim=-1)
        logits = (q_dim @ p_dim.t()) / self.temperature
        contrastive = F.cross_entropy(logits, target)

        q_small = F.normalize(qry[:, :dim], p=2, dim=-1)
        p_from_big = F.normalize(self._project_to_dim(model, pos, dim, src_dim=bigger_dim), p=2, dim=-1)
        cosine_map_1 = q_small @ p_from_big.t()

        q_from_big = F.normalize(self._project_to_dim(model, qry, dim, src_dim=bigger_dim), p=2, dim=-1)
        p_small = F.normalize(pos[:, :dim], p=2, dim=-1)
        cosine_map_2 = q_from_big @ p_small.t()

        l1_consistency = F.l1_loss(cosine_map_1, cosine_map_2)
        return contrastive, l1_consistency

    def _resolve_dims(self, full_dim: int) -> List[int]:
        valid_dims = [d for d in self.nested_dims if d <= full_dim]
        if full_dim not in valid_dims:
            valid_dims.append(full_dim)
        return sorted(set(valid_dims))

    def _parse_dim_weight_map(self, spec) -> Dict[int, float]:
        if not spec:
            return {}
        out: Dict[int, float] = {}
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            dim_str, weight_str = item.split(":", 1)
            out[int(dim_str.strip())] = float(weight_str.strip())
        return out

    def _resolve_selected_stage_ids(self, stage_pairs: List[Tuple[int, int]]) -> List[int]:
        max_idx = len(stage_pairs) - 1
        phase = self.phase.strip().upper()
        if phase == "ALL":
            return list(range(len(stage_pairs)))
        tokens = [tok.strip() for tok in phase.replace(";", ",").split(",") if tok.strip()]
        selected_ids: List[int] = []
        for token in tokens:
            if token.isdigit():
                selected_ids.append(int(token))
            elif token.isalpha():
                selected_ids.append(ord(token[-1]) - ord("A"))
        selected_ids = sorted({idx for idx in selected_ids if 0 <= idx <= max_idx})
        return selected_ids or [0]

    def _parse_pair_weight_map(self, spec) -> Dict[Tuple[int, int], float]:
        if not spec:
            return {}
        out: Dict[Tuple[int, int], float] = {}
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

    @staticmethod
    def _resolve_pair_weight(weight_map: Dict[Tuple[int, int], float], src_dim: int, dst_dim: int, default: float = 1.0) -> float:
        return float(weight_map.get((src_dim, dst_dim), default))

    def _parse_projection_pairs(self, spec: str) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        if not spec:
            return out
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            if "->" in item:
                src_str, dst_str = item.split("->", 1)
            else:
                src_str, dst_str = item.split(":")
            src_dim, dst_dim = int(src_str.strip()), int(dst_str.strip())
            if src_dim <= dst_dim:
                raise ValueError(f"Invalid projection pair {src_dim}->{dst_dim}.")
            out.append((src_dim, dst_dim))
        return out

    def forward(self, model_trainer, input_data: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        model = model_trainer.model
        qry_full = model.encode_input(input_data["qry"])[0]
        pos_full = model.encode_input(input_data["pos"])[0]
        if self.world_size > 1:
            qry_full = self._dist_gather_tensor(qry_full)
            pos_full = self._dist_gather_tensor(pos_full)

        valid_dims = self._resolve_dims(qry_full.size(-1))
        target = self._build_contrastive_target(qry_full, pos_full)
        desc_dims = sorted(valid_dims, reverse=True)
        if self.projection_spec:
            vset = set(valid_dims)
            stage_pairs = [(s, d) for s, d in self._parse_projection_pairs(self.projection_spec) if s in vset and d in vset]
        else:
            stage_pairs = [(s, d) for s in desc_dims for d in desc_dims if s > d]
        stage_pairs = list(dict.fromkeys(stage_pairs))

        metrics: Dict[str, Tensor] = {}
        if not stage_pairs:
            align_ce, align_l1 = self._cross_alignment_l1(model, qry_full, pos_full, target, desc_dims[0], desc_dims[0])
            weighted_align = align_ce + self.full_dim_l1_weight * align_l1
            metrics["loss"] = weighted_align
            metrics["total_loss"] = weighted_align.detach()
            metrics["contrastive_loss"] = weighted_align.detach()
            metrics["align_loss"] = weighted_align.detach()
            metrics["orthogonal_loss"] = torch.zeros_like(weighted_align).detach()
            metrics["spectrum_kl_loss"] = torch.zeros_like(weighted_align).detach()
            metrics["spectrum_kl_weight"] = torch.tensor(0.0, device=weighted_align.device)
            return metrics

        losses, align_losses, orth_losses = [], [], []
        for idx in self._resolve_selected_stage_ids(stage_pairs):
            teacher_dim, student_dim = stage_pairs[idx]
            align_ce, align_l1 = self._cross_alignment_l1(model, qry_full, pos_full, target, student_dim, teacher_dim)
            l1_weight = self.dim_align_l1_weights.get(student_dim, self.align_l1_weight)
            weighted_align = align_ce + l1_weight * align_l1
            projection_weight = self._resolve_pair_weight(self.projection_weights, teacher_dim, student_dim, 1.0)

            if hasattr(model, "matryoshka_proj_bank") and not getattr(model.matryoshka_proj_bank, "use_orthogonal_parametrization", False):
                base_orth = model.matryoshka_proj_bank.orthogonality_loss(src_dim=teacher_dim, dst_dim=student_dim)
                orth_pair_weight = self._resolve_pair_weight(self.orthogonal_pair_weights, teacher_dim, student_dim, 1.0)
                orth_loss = orth_pair_weight * base_orth
            else:
                orth_loss = torch.zeros_like(weighted_align)
            losses.append(projection_weight * weighted_align + self.orthogonal_weight * projection_weight * orth_loss)
            align_losses.append(weighted_align)
            orth_losses.append(orth_loss)

        final_loss = torch.stack(losses).mean()
        if hasattr(model, "matryoshka_proj_bank"):
            final_loss = final_loss + model.matryoshka_proj_bank.zero_weight_ddp_param_touch()
        metrics["loss"] = final_loss
        metrics["total_loss"] = final_loss.detach()
        metrics["contrastive_loss"] = torch.stack(align_losses).mean()
        metrics["align_loss"] = metrics["contrastive_loss"].detach()
        metrics["orthogonal_loss"] = torch.stack(orth_losses).mean().detach()
        metrics["spectrum_kl_loss"] = torch.zeros_like(final_loss).detach()
        metrics["spectrum_kl_weight"] = torch.tensor(0.0, device=final_loss.device)
        return metrics
