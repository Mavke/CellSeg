import torch
import torch.nn as nn


class PantopicQuality(nn.Module):
    
    def __init__(self):
        super(PantopicQuality, self).__init__()
    
    def forward(self, gt, pred):
        tp = 0
        global_iou = 0
        for label in torch.unique(gt):
            gt_segment = gt == label
            pred_labels = torch.unique(pred * gt_segment)

            for pred_label in pred_labels:
                pred_segement = pred == pred_label
                iou = torch.sum(gt_segment * pred_segement) / torch.sum(gt_segment + pred_segement)
                if iou > 0.5:
                    gt[gt_segment] = 0
                    pred[pred_segement] = 0
                    tp += 1
                    global_iou += iou
        
        fn = torch.unique(gt).shape[0]
        fp = torch.unique(pred).shape[0]

        return iou / (tp + 0.5 * (fn + fp))
            