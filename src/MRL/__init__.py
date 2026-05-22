from .adaptive_inference import route_and_truncate_query
from .base_mrl import MatryoshkaContrastiveLoss
from .ese import ESELoss
from .adaptive_matryoshka import AdaptiveMatryoshkaStage1Loss, AdaptiveRouterLoss
from .adaptive_projection_only import AdaptiveProjectionOnlyStage1Loss
from .adaptive_laplacian_only import AdaptiveLaplacianOnlyStage1Loss

criterion_dict = {
    "mrl": MatryoshkaContrastiveLoss,
    "ese": ESELoss,
    "adaptive_mrl_stage1": AdaptiveMatryoshkaStage1Loss,
    "adaptive_router": AdaptiveRouterLoss,
    "adaptive_mrl_projection_only": AdaptiveProjectionOnlyStage1Loss,
    "adaptive_mrl_laplacian_only": AdaptiveLaplacianOnlyStage1Loss,
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_dict.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_dict[args.kd_loss_type](args)