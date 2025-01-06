import torch
import torch.nn as nn

from evaluation.utils import one_hot_encoder

class FocalTverskyLoss(nn.Module):

    def __init__(self, n_classes: int, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-8):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.n_classes = n_classes

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        pred = torch.softmax(pred, dim=1)
        if pred.size() != gt.size():
            gt = one_hot_encoder(gt, self.n_classes)
            assert pred.size() == gt.size(), 'prediction {} & ground_truth {} size do not match'.format(pred.size(), gt.size())

        tversky_loss = 0

        for cl in range(self.n_classes):
            tversky_index = self.get_tversky_index(pred[:, cl, :, :], gt[:, cl, :, :])

            tversky_loss += (1-tversky_index)**self.gamma
        
        return (tversky_loss / self.n_classes)
    
    def get_tversky_index(self, pred, gt):
        true_positives = (pred * gt).sum()
        false_positives = self.alpha * torch.sum((1 - pred) * gt)
        false_negative = self.beta * torch.sum(pred * (1 - gt))

        tversky_index = (true_positives + self.smooth) / (true_positives + false_positives + false_negative + self.smooth)

        return tversky_index
