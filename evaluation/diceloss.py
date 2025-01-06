import torch
import torch.nn as nn
from evaluation.utils import one_hot_encoder

class DiceLoss(nn.Module):
    def __init__(self, n_classes, special_eval = False):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.special_eval = special_eval

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
            if self.special_eval:
                inputs[:,0,:,:] = inputs[:,0,:,:]
                inputs[:,1,:,:] = 1 - inputs[:,0,:,:]
                inputs = inputs[:,0:2,:,:]
        if inputs.size() != target.size():
            target = one_hot_encoder(target, self.n_classes)
        if weight is None:
            weight = [1] * self.n_classes
        
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(dice.item())
            loss += dice * weight[i]
            
        return loss / self.n_classes, class_wise_dice