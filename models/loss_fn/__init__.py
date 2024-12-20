from monai.losses.dice import DiceLoss
from .unified_loss_fn import UnifiedLoss
from .focal_loss_fn import FocalLoss

def load_loss_fn(loss_fn='dice', reduction='mean'):
    if loss_fn == 'unified':
        return UnifiedLoss(reduction=reduction)
    elif loss_fn == 'dice':
        return DiceLoss(reduction=reduction)
    elif loss_fn == 'focal':
        return FocalLoss()