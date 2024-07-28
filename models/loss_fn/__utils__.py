from monai.losses.dice import DiceLoss
from .unified_loss_fn import UnifiedLoss

def get_loss_fn(imbalance=False, reduction='mean'):
    if imbalance:
        return UnifiedLoss(reduction=reduction)
    else:
        return DiceLoss(reduction=reduction)