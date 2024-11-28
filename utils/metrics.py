# Jaccard Index(IoU), Dice Coefficient(F1-Score), Precision, Recall
import torch

def base_metrics(outputs, targets):
    '''
    return TP, FP, TN, FN
    '''
    # to binary tensor
    outputs = torch.where((outputs >= 0.5), 1., 0.)
    
    # reshape to sum (consider Complex dimension to simplify)
    outputs = outputs.view(outputs.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    
    # batch-wise (N, ), torch.sum(tensor, dim=d)
    '''
    만약, d가 1이고, tensor가 (4, 3) shape이라면 (4, 0) + (4, 1) + (4, 2)의 텐서들이 더해진다.
    즉, 차원을 인덱스 삼아 다 더한다.
    '''
    TP = torch.sum(torch.where((outputs == targets) & (outputs == 1.), 1, 0), 1)
    FP = torch.sum(torch.where((outputs != targets) & (outputs == 1.), 1, 0), 1)
    TN = torch.sum(torch.where((outputs == targets) & (outputs == 0.), 1, 0), 1)
    FN = torch.sum(torch.where((outputs != targets) & (outputs == 0.), 1, 0), 1)
    
    return TP, FP, TN, FN

def get_metrics(outputs, targets, metrics):
    TP, FP, TN, FN = base_metrics(outputs, targets)
    print(TP, FP, TN, FN)
    metrics_dict = {}
    TP = torch.where(TP == 0., 1e-6, TP)
    
    if 'IoU' in metrics:
        iou = TP / (TP + FP + FN); metrics_dict['IoU'] = iou.detach().cpu().tolist()
    if 'Dice' in metrics:
        dice = 2*TP / (2*TP + FP + FN); metrics_dict['Dice'] = dice.detach().cpu().tolist()
    if 'Precision' in metrics:
        prec = TP / (TP + FP); metrics_dict['Precision'] = prec.detach().cpu().tolist()
    if 'Recall' in metrics:
        recall = TP / (TP + FN); metrics_dict['Recall'] = recall.detach().cpu().tolist()
    
    return metrics_dict

if __name__ == '__main__':
    # outputs = torch.tensor([[0.8, 0.2, 0.3, 0.6], [0.65, 0.14, 0.5, 0.2]])
    outputs = torch.tensor([[[0.8, 0.2], [0.3, 0.6]], [[0.65, 0.14], [0.5, 0.2]]])
    # targets = torch.tensor([[1., 0., 0., 0.], [1., 1., 1., 0.]])
    targets = torch.tensor([[[1., 0.], [0., 0.]], [[1., 1.], [1., 0.]]])
    print(f'Original Tensor shape: {outputs.shape}')
    
    print(*base_metrics(outputs, targets), sep='\n')
    metrics_dict = get_metrics(outputs, targets, ['IoU', 'Dice', 'Precision', 'Recall'])
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")