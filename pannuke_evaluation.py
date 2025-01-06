import os, torch, copy, argparse, cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from torch.utils.data import DataLoader

from datasets.pannuke_dataset import PannukeDataset


from tqdm import tqdm

from model.myformer import MyFormer


from evaluation.metric import Metrics, MetricsPerClass
from evaluation.panoptic import Pantopic
from reference_models.cellvit.models.segmentation.cell_segmentation.cellvit import CellViTSAM


from post_processing.post_processing import PostProcessor
from post_processing.pannuke_pq import get_fast_pq , remap_label

CLASS_DICT = {
            0: "Background",
            5: 'Neoplastic cells',
            1: 'Inflammatory cells',
            2: 'Connective/Soft tissue cells',
            3: 'Dead Cells',
            4: 'Epithelial cells'
        }

COLOR_DICT = {
    1: [255, 0, 0],
    2: [34, 221, 77],
    3: [35, 92, 236],
    4: [254, 255, 0],
    5: [255, 159, 68],
}


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class FoldInference():

    def __init__(self, args):
        self.args = args
        self.base_dir = args.base_dir
        self.model_file = args.model_file
        self.model_name = args.model_name
        self.fold = args.fold
        self.test_split = args.test_split
        self.save_result = args.save_result
        self.save_path = args.save_path

        self.n_classes = 6
        self.n_classes_metric_eval = 6
        batch_size = 1
        self.image_size = (256, 256)
        channel_dim = 3

        db_val = PannukeDataset(args.test_split, args.fold, self.image_size, "test")
        self.valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=1)

        self.dict_key_np = 'nuclei_binary_map'
        self.dict_key_hv = 'hv_map'


        if args.model_type == 'my_model':
            self.model = MyFormer(self.image_size, self.n_classes, window_size=16, pretrained_transformer='./pretrained_models/swin_tiny_patch4_window16_256.pth', channel_dim=channel_dim)
            self.dict_key_nc = 'class'
        else:
            self.model = CellViTSAM(args.base_dir + args.model_file, num_nuclei_classes=6, num_tissue_classes=0, vit_structure='SAM-B')
            self.dict_key_nc = 'nuclei_type_map'

    
        model_path = args.base_dir + args.model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()

    def run_inference(self):

        pantopic = Pantopic()
        post_processor = PostProcessor()
        metric_evaluator = Metrics(2)
        metric_evaluator_classes = MetricsPerClass(self.n_classes, self.device)

        metrics_complete_set = {'accuarcy': 0, 'precision': 0, 'IuO': 0, 'dice': 0, 'recall': 0}
        metric_instance = {'dice': 0, 'precision': 0, 'recall': 0}
        metric_instance_cl = [np.zeros((len(self.valloader), 3)) for x in range(1, self.n_classes)]
        pq_values = []
        nuclei_type_pq = []


        for idx, batch in enumerate(tqdm(self.valloader)):

            images, labels, instance_mask = batch['image'].to(self.device), batch['label'].to(self.device), batch['instance_mask'].to(self.device)
            class_mask_gt = batch['class_mask'].to(self.device).squeeze()
            class_mask = torch.argmax(class_mask_gt, dim=0).squeeze().cpu().detach().numpy()
            
            i_mask = instance_mask.squeeze().cpu().detach().numpy()
            t = ((i_mask - np.min(i_mask)) / (np.max(i_mask) - np.min(i_mask))) * 255
            cv2.imwrite(self.save_path + f'/t{idx}.jpg', t)
            res = self.model(images)

            out = res[self.dict_key_np]
            distance_map = res[self.dict_key_hv]
    
            nuclei_classes_pred  = res[self.dict_key_nc].squeeze()
            nuclei_classes = torch.softmax(copy.copy(nuclei_classes_pred), dim=0)
            nuclei_classes = torch.argmax(nuclei_classes, dim=0).squeeze().cpu().detach().numpy()


            temp = torch.softmax(out[0, :], dim=0)
            temp = torch.argmax(temp, dim=0)

            blb = temp.squeeze().cpu().detach().numpy()
            d_map = distance_map.squeeze().cpu().detach().numpy()

            test = np.concatenate((np.expand_dims(nuclei_classes, -1), np.expand_dims(blb, -1), np.expand_dims(d_map[0], -1), np.expand_dims(d_map[1], -1)), axis=-1)
            true_instance_mask = remap_label(copy.deepcopy(instance_mask).cpu().detach().numpy().squeeze())

            pred = post_processor.process(test, nr_types=5)

            #get nuclei pq
            if len(np.unique(true_instance_mask)) == 1:
                pq = np.nan
            else:
                [_,_,pq], _ = get_fast_pq(true_instance_mask, remap_label(pred[0]))
            pq_values.append(pq)

            # get nuclei type pq
            nuclei_classes_pred = torch.softmax(nuclei_classes_pred, dim = 0).cpu().detach().numpy() > 0.5
            for j in range(1, self.n_classes):
                nuclei_classes_j_pred = remap_label(
                    nuclei_classes_pred[j, ...]) 
                gt_nuclei_instance_class = remap_label(
                    class_mask_gt[j, ...].cpu().detach().numpy()
                )

                if len(np.unique(gt_nuclei_instance_class)) == 1:
                    pq_tmp = np.nan
                    dq_tmp = np.nan
                    sq_tmp = np.nan
                else:
                    [dq_tmp, sq_tmp, pq_tmp], _ = get_fast_pq(
                        nuclei_classes_j_pred,
                        gt_nuclei_instance_class,
                        match_iou=0.5,
                    )
                nuclei_type_pq.append(pq_tmp)

           
            class_map = np.zeros((256,256))
       
            for label in np.unique(pred[0]):
                if label == 0:
                    continue
                mask = pred[0] == label
                if label not in pred[1].keys():
                    continue
                class_map[mask] = pred[1][label]['type']
            
            if self.save_result and idx < 10:
                print('Saving images')
                image = images.squeeze().cpu().detach().numpy()

                image = images.squeeze().cpu().detach().numpy()
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                image = (image * 255).astype(np.uint8)

                image = np.swapaxes(image, 0, 2)
                image = np.swapaxes(image, 0, 1)
                structuring_element = np.ones((3, 3))
              
                combined_image = np.copy(image)

                for label_pred in np.unique(pred[0]):
                    if label_pred == 0:
                        continue
                    erosion_mask = ndimage.binary_erosion(pred[0] == label_pred, structuring_element)
                    dilation_mask = ndimage.binary_dilation(pred[0] == label_pred, structuring_element)
                    edge = np.asarray(dilation_mask, dtype=np.int16) - np.asarray(erosion_mask, dtype=np.int16)

                    instace_props = pred[1][label_pred]
                    if instace_props['type'] == 0:
                        continue
                    combined_image[edge != 0] = COLOR_DICT[instace_props['type']]
                cv2.imwrite(self.save_path + f'/pred{idx}.jpg', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))


            metric_evaluator(temp, labels[0, :] != 0)



            #segmentation quality for the classes, instances not considered
            metric_evaluator_classes(copy.deepcopy(class_map), class_mask)


            np_instance_mask_gt = copy.deepcopy(instance_mask).cpu().detach().numpy()
            class_map_gt = labels[0, :].cpu().detach().numpy()
    
            metrics_, metric_class = pantopic.get_metrics(copy.deepcopy(pred[0]), np_instance_mask_gt, class_map, class_mask, self.n_classes)

            metric_instance['dice'] += metrics_['dice']
            metric_instance['precision'] += metrics_['precision']
            metric_instance['recall'] += metrics_['recall']

            for i in range(1, self.n_classes):
                metric_instance_cl[i-1][idx, 0] = metric_class[i-1]['dice']
                metric_instance_cl[i-1][idx, 1] = metric_class[i-1]['precision']
                metric_instance_cl[i-1][idx, 2] = metric_class[i-1]['recall']


            if self.save_result and idx < 10:
                print('Saving images')
                image = images.squeeze().cpu().detach().numpy()

                image = images.squeeze().cpu().detach().numpy()
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                image = (image * 255).astype(np.uint8)

                image = np.swapaxes(image, 0, 2)
                image = np.swapaxes(image, 0, 1)
                structuring_element = np.ones((3, 3))
              
                combined_image = np.copy(image)
                combined_image_gt = np.copy(image)


                instance_mask = np.squeeze(instance_mask.cpu().detach().numpy())
                
                for label_gt in np.unique(instance_mask):
                    if label_gt == 0:
                        continue

                    mask_label = instance_mask == label_gt

                    instance_class = np.unique(class_mask * mask_label).sum()
                    
                    erosion_mask = ndimage.binary_erosion(mask_label, structuring_element)
                    dilation_mask = ndimage.binary_dilation(mask_label, structuring_element)

                    edge = np.asarray(dilation_mask, dtype=np.int16) - np.asarray(erosion_mask, dtype=np.int16)

                    combined_image_gt[edge != 0] = COLOR_DICT[instance_class]

                cv2.imwrite(self.save_path + f'/gt{idx}.jpg', cv2.cvtColor(combined_image_gt, cv2.COLOR_RGB2BGR))
        
        
        n = len(self.valloader)

        class_metrics_dict = metric_evaluator_classes.get_mean_values()
        
        for k, v in metric_instance.items():
            metric_instance[k] /= n

        metrics_complete_set = metric_evaluator.get_mean_values()

        self.save_metics_as_csv(metric_evaluator_classes.class_metrics_dict, args.base_dir)

        with open(args.base_dir + '/results.txt', 'a') as f:
            f.write(f'{self.model_name} \n')
            f.write(f'Test split : {args.test_split} \n')
            f.write(f'Image size : {self.image_size} \n')
            
            for k, v in metrics_complete_set.items():
                f.write(f'{k} : {v} \n')
            
            f.write('\n Instance metrics \n')
            
            for k, v in metric_instance.items():
                f.write(f'{k} : {v} \n')

            for cl in range(1, self.n_classes):
                f.write(f'Class {cl} \n')
                f.write(f'Dice : {class_metrics_dict[cl-1]["dice"]} \n')
                f.write(f'Precision : {class_metrics_dict[cl-1]["precision"]} \n')
                f.write(f'Recall : {class_metrics_dict[cl-1]["recall"]} \n')
                f.write(f'Accuracy : {class_metrics_dict[cl-1]["accuarcy"]} \n')
                f.write(f'IoU : {class_metrics_dict[cl-1]["iou"]} \n')
    
            for cl in range(1, self.n_classes):
                f.write(f'Class {CLASS_DICT[cl]} \n')
                f.write(f'Dice : {np.nanmean(metric_instance_cl[cl-1][:, 0])} \n')
                f.write(f'Precision : {np.nanmean(metric_instance_cl[cl-1][:, 1])} \n')
                f.write(f'Recall : {np.nanmean(metric_instance_cl[cl-1][:, 2])} \n')

            f.write(f'PQ : {np.nanmean(pq_values)} \n')
            f.write(f'Nuclei type PQ : {np.nanmean(nuclei_type_pq)} \n')
            f.write('\n')
            f.close()
     
    def save_metics_as_csv(self, dict, path):
        for i in range(1, 6):
            class_metrics = dict[i-1]
            cols = class_metrics.keys()
            values = class_metrics.values()
            df = pd.DataFrame(values).transpose()
            df.columns = cols

            df.to_csv(path + f'/class_{i}_metrics.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='H2Former testing')

    parser.add_argument('--base_dir', type=str, default='./')
    parser.add_argument('--model_file', type=str, default='H2Former130.pth')
    parser.add_argument('--model_name', type=str, default='H2Former')
    parser.add_argument("--fold", type=int, default=3)

    parser.add_argument('--test_split', type=str, default='./')
    parser.add_argument('--save_result', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./predictions')
    parser.add_argument('--model_type', type=str, default="my_model")
    args = parser.parse_args()

    evaluator = FoldInference(args)
    evaluator.run_inference()
