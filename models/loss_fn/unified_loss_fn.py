import torch
from torch import nn
from monai.losses.dice import DiceLoss

class WeightedCrossEntropyLoss(nn.Module):
    '''
    # device 바꿔주기 잊지 말기 -> 쓰이는 Tensor가 없어서 GPU에 옮기는 작업이 필요 없을 듯
    # spine의 비율이 통상적으로 4~7% 정도 나옴
    # 한 20배 곱해주는 게 좋을 듯
    # 너무 커서 학습 과정에서 NaN으로 나왔다고 판단
    # 이에 대해 log 값에 대한 Threshold를 지정 minus_inf_threshold
    # 각 픽셀의 loss를 sum에서 mean으로 변경하여 loss 값을 도출
    # weight는 20배에서 3배로 변경
    '''
    def __init__(self, weight:float=20., reduction='mean'):
        
        if weight <= 1.0:
            raise AssertionError("Positive가 imbalance해서 만든 함수인데 왜 positive에 대한 weight가 더 적은 것인지 다시 확인해볼 것")
        super().__init__()
        
        self.weight = weight
        self.reduction = reduction
    
    def transform_log(self, outputs, minus_inf_threshold=-5, eps=1e-7):
        # Epsion의 역할은 log(0)이면, 무한대이기 때문에 이를 방지하도록 함.
        input_ = torch.log(outputs + eps)
        condition = (input_ < minus_inf_threshold)
        # 또한, loss가 너무 커지지 않도록 threshold 값을 통해 조절
        return torch.where(condition, minus_inf_threshold, input_)
    
    def forward(self, outputs, targets):
        # outputs는 모델에서 마지막 Acti를 거치고 나오기 때문에 여기서는
        # 별도로 또 Activation을 해줄 필요는 없음.
        
        # target이 아닌 곳은 0이기 때문에 'target이 1인 것만 가져와서 계산'
        # 같은 번거로운 과정은 필요 없음.
        
        outputs = outputs.view(outputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        
        pos_log_outputs = self.transform_log(outputs)
        neg_log_outputs = self.transform_log(1-outputs)
        
        positive_loss = -(targets * pos_log_outputs) * self.weight
        negative_loss = -((1-targets)*neg_log_outputs)
        
        loss = positive_loss + negative_loss
        loss = torch.mean(loss, 1) # Batch 차원을 제외한 각 픽셀들의 loss 모두 더하기 (N, )
        
        if self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'none':
            # batch 단위로 리턴
            return loss

class UnifiedLoss(nn.Module):
    '''
    BU-Net의 imbalance를 해결하기 위해
    Dice Loss와 Weighted Cross Entropy를 합친 것
    '''
    def __init__(self, weight:float=3., reduction='mean', return_total_loss=True):
        super().__init__()
        
        # Positive(1) = Spine, Negative(0) = Background
        self.DLC = DiceLoss(reduction=reduction)
        self.WCE = WeightedCrossEntropyLoss(weight=weight,
                                            reduction=reduction)
        self.return_total_loss = return_total_loss
    
    def forward(self, outputs, targets):
        loss_1 = self.DLC(outputs, targets)
        loss_2 = self.WCE(outputs, targets)
        
        if self.return_total_loss:
            return loss_1+loss_2
        else:
            loss_dict = dict(dice_loss=loss_1,
                            wce_loss=loss_2,
                            total_loss=(loss_1+loss_2))
            
            return loss_dict


if __name__ == '__main__':
    loss_fn = UnifiedLoss()
    a = torch.zeros(16, 1, 144, 144) + 0.9
    b = torch.ones(16, 1, 144, 144)
    loss = loss_fn(a, b)
    print(loss)