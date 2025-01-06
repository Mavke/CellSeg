import statistics
import torch
import torch.nn as nn
import numpy as np

class Metrics(nn.Module):
    def __init__(self, n_classes):
        super(Metrics, self).__init__()
        self.n_classes = n_classes
        self.dice = 0
        self.accuarcy = 0
        self.precision = 0
        self.iou = 0
        self.recall = 0
        self.entries = 0

    def forward(self, pred, gt):

        for i in range(0,self.n_classes-1):
            pred_label_i = pred == i+1
            gt_label_i = gt == i+1

            TP = torch.sum(pred_label_i * gt_label_i)
            TN = torch.sum((pred_label_i==0) * (gt_label_i==0))

            FN = torch.sum((pred_label_i==0) * (gt_label_i==1))
            FP = torch.sum((pred_label_i==1) * (gt_label_i==0))
            
            if TP.item() == 0:
                TP = 1
            recall = TP / (TP + FN)
            accuarcy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            dice = 2 * TP / (2*TP + FP + FN)
            iou = TP / (TP + FN + FP)

            self.update_values(dice = dice, accuarcy=accuarcy, precision = precision, iou=iou, recall=recall)

    def update_values(self, dice: float, accuarcy: float, precision: float, recall: float,  iou: float):
        self.dice += dice
        self.accuarcy += accuarcy
        self.precision += precision
        self.iou += iou
        self.recall += recall
        self.entries +=1
    
    def get_mean_values(self):
        mean_values = {}
        mean_values['dice'] = self.dice / self.entries
        mean_values['accuarcy'] = self.accuarcy / self.entries
        mean_values['precision'] = self.precision / self.entries
        mean_values['iou'] = self.iou / self.entries
        mean_values['recall'] = self.recall / self.entries

        return mean_values
    
    def as_string(self):
        metric_string = f'dice : {self.dice} | accuarcy : {self.accuarcy} |precision : {self.precision} | iuo : {self.iou} | recall : {self.recall}'
        
        return metric_string
       

class MetricsPerClass(nn.Module):
    def __init__(self, n_classes, device):
        super(MetricsPerClass, self).__init__()
        self.n_classes = n_classes
        self.device = device
        self.class_metrics_dict = [{'dice': [], 'precision': [], 'recall': [], 'accuarcy' : [], 'iou' : []} for x in range(1, self.n_classes)]
        self.updates = 0
    
    def _one_hot_encoding(self, data):
        one_hot = torch.zeros((self.n_classes, data.shape[0], data.shape[1])).to(self.device)
        for i in range(self.n_classes):
            one_hot[i,:] = data == i 

        return one_hot

    def forward(self, pred, gt):
        self.updates += 1
        
        for i in range(1,self.n_classes):
            pred_label_i = pred == i
            gt_label_i = gt == i

            #class is non existent
            if pred_label_i.sum() == 0 and gt_label_i.sum() == 0:
                continue

            TP = np.sum(pred_label_i * gt_label_i)
            TN = np.sum((pred_label_i==0) * (gt_label_i==0))

            FN = np.sum((pred_label_i==0) * (gt_label_i==1))
            FP = np.sum((pred_label_i==1) * (gt_label_i==0))
            
            
            if np.unique(gt_label_i).shape[0] == 1 and np.unique(gt_label_i) == 0:
                FN = 0
        
            if np.unique(pred_label_i).shape[0] == 1 and np.unique(pred_label_i) == 0:
                FP = 0

            recall = TP / (TP + FN + 1e-8)
            accuarcy = (TP + TN) / (TP + TN + FP + FN+ 1e-8)
            precision = TP / (TP + FP+ 1e-8)
            dice = 2 * TP / (2*TP + FP + FN+ 1e-8)
            iou = TP / (TP + FN + FP+ 1e-8)

            if np.isnan(dice):
                dice = 0

            self.class_metrics_dict[i-1]['dice'].append(dice)
            self.class_metrics_dict[i-1]['precision'].append(precision)
            self.class_metrics_dict[i-1]['recall'].append(recall)
            self.class_metrics_dict[i-1]['iou'].append(iou)
            self.class_metrics_dict[i-1]['accuarcy'].append(accuarcy)

    def get_mean_values(self):
        mean_values = {}
        for i in range(1, self.n_classes):
            mean_values[i-1] = {k : (statistics.mean(v)) for k,v in self.class_metrics_dict[i-1].items()}
            
        return mean_values