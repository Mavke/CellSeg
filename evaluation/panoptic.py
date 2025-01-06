import numpy as np


def get_center_of_masses(masks: np.ndarray, label):
    com = np.zeros((1, 2))
    x, y = np.where(masks == label)
    center_x = np.mean(x)
    center_y = np.mean(y)
    com[0, 0] = center_x
    com[0, 1] = center_y

    return com


class Pantopic():

    def __init__(self):
        super(Pantopic, self).__init__()

    def get_metrics(self, pred, gt, class_map_pred, class_map_gt, n_classes=6):   
        gt = gt.squeeze()
        pred = pred.squeeze()
        tp_d, fp_d, fn_d, matched_pred_types, matched_gt_types, unmatched_true, unmatched_pred = self.evaluate_instances(pred, gt, class_map_pred, class_map_gt)

        results_d = {'dice': 0, 'precision': 0, 'recall': 0}
        result_class = [{'dice': np.NAN, 'precision': np.NAN, 'recall': np.NAN} for x in range(1, n_classes)]

        for cl in range(1, n_classes):
            # if no instances of this class are present 
            if np.sum(matched_gt_types == cl) == 0 and np.sum(matched_pred_types == cl) == 0:
                continue
 
            type_samples = (matched_pred_types == cl) | (matched_gt_types == cl)
            matched_gt_types_c = matched_gt_types[type_samples]
            matched_pred_types_c = matched_pred_types[type_samples]
             
            tp_c = ((matched_gt_types_c == cl) & (matched_pred_types_c == cl)).sum()
            tn_c = ((matched_gt_types_c != cl) & (matched_pred_types_c != cl)).sum()
            fp_c = ((matched_gt_types_c != cl) & (matched_pred_types_c == cl)).sum()
            fn_c = ((matched_gt_types_c == cl) & (matched_pred_types_c != cl)).sum()   

            fn_dc = np.sum(unmatched_true == cl)
            fp_dc = np.sum(unmatched_pred == cl)
            f1_c = (2 * (tp_c + tn_c)) / (2 * (tp_c + tn_c) + 2 * fp_c + 2 * fn_c + fp_dc + fn_dc + 1e-8)

            p_c = (tp_c + tn_c) / (tp_c + tn_c + 2 * fp_c + fp_dc+ 1e-8)

            r_c = (tp_c + tn_c) / (tp_c + tn_c + 2 * fn_c + fn_dc+ 1e-8)

            result_class[cl-1]['dice'] = f1_c
            result_class[cl-1]['precision'] = p_c
            result_class[cl-1]['recall'] = r_c

        results_d['dice'] = 2 * tp_d / ((2*tp_d + fp_d + fn_d) + 1e-8)
        results_d['precision'] = tp_d / (tp_d + fp_d + 1e-8)
        results_d['recall'] = tp_d / (tp_d + fn_d + 1e-8)

        return results_d, result_class

    def evaluate_instances(self, pred, gt, class_map_pred, class_map_gt):


        matched_gt_types = []
        matched_pred_types = []
        labels = np.unique(gt)[1:]
        tp = 0

        for label in labels:
            gt_seg = gt == label
            center_gt_seg = get_center_of_masses(gt_seg, 1)
            pred_seg = np.zeros((pred.shape))
            pred_seg = pred[gt_seg]
            overlapping_labels = np.unique(pred_seg)

            for pred_label in overlapping_labels:
                pred_instance_seg = np.zeros((pred.shape))
                pred_instance_seg = pred == pred_label
                center_pred_seg = get_center_of_masses(pred_instance_seg, 1)

                if np.sum((center_pred_seg - center_gt_seg)**2) < 12**2:
                    class_gt = np.unique(class_map_gt[gt_seg])[-1]
                    class_pred = np.unique(class_map_pred[pred_instance_seg])[-1]

                    matched_gt_types.append(class_gt)
                    matched_pred_types.append(class_pred)

                    tp += 1
                    gt[gt_seg] = 0
                    pred[pred_instance_seg] = 0

        if np.unique(gt).shape[0] == 1 and np.unique(gt) == 0:
            fn = 0
            unmatched_true = 0
        else:
            fn = (np.unique(gt) != 0).sum()
            unmatched_true = self._retrieve_unmatched_classes(gt, class_map_gt)
    
        if np.unique(pred).shape[0] == 1 and np.unique(pred) == 0:
            fp = 0
            unmatched_pred = 0
        else:
            fp = (np.unique(pred) != 0).sum()
            unmatched_pred = self._retrieve_unmatched_classes(pred, class_map_pred)

        return tp, fp, fn, np.asarray(matched_pred_types), np.asarray(matched_gt_types), unmatched_true, unmatched_pred

    def _retrieve_unmatched_classes(self, instance_map, class_map):
        unmatched_cl = []
        for instance in (np.unique(instance_map)):
            if instance == 0:
                continue
            cl = np.unique(class_map * (instance_map == instance)).sum()
            unmatched_cl.append(cl)
        
        return np.asarray(unmatched_cl)