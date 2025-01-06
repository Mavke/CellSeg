import torch
import cv2
import albumentations as A

from torch.utils.data import Dataset
from datasets.data_set_prep import *
from PIL import Image



class SporulationDataset(Dataset):

    def __init__(self, data_path: str, output_size: int, split: str, sampel_list_path: str, model_type: str = 'instance'):
        self.data_path = data_path
        self.w, self.h = output_size
        self.model_type = model_type


        self.split = split
        self.sample_list = open(sampel_list_path).readlines()
        self.transform = A.Compose([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), always_apply=True),
                A.Rotate((90,90), interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomScale(scale_limit=[0.5, 0.5], p = 0.15),
                A.ElasticTransform(sigma=25, alpha=0.5, alpha_affine=15, p=0.2)], is_check_shapes=False)
        self.test_transform = A.Compose([A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), always_apply=True)], is_check_shapes=False)
        
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        sample_id = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_path, (sample_id + '_img.tif'))
        mask_path = os.path.join(self.data_path, (sample_id + '_mask.tif'))

        img = Image.open(img_path)
        mask = Image.open(mask_path) 

        img = np.array(img, dtype=np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = np.asarray(img, dtype=np.uint8) 
        mask = np.array(mask, dtype=np.float32)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        #remove overlap between instances for semantic models
        if self.model_type == 'semantic':
            overlap = get_overlapping_edges(mask)
            mask = mask * (1 - overlap)
        
        if self.split == 'train':
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

            distance_map_h, distance_map_v = get_distance_maps_single_mask(mask)
            mask = cropping_center(mask, (80,80))
            distance_map_h = cropping_center(distance_map_h, (80,80))
            distance_map_v = cropping_center(distance_map_v, (80,80))
            distance_map = np.concatenate((distance_map_h.squeeze()[:,:,np.newaxis], distance_map_v.squeeze()[:,:,np.newaxis]), axis=-1)
        else:
            transformed = self.test_transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

            distance_map_h, distance_map_v = get_distance_maps_single_mask(mask)

            mask = cropping_center(mask, (80,80))
            distance_map_h = cropping_center(distance_map_h, (80,80))
            distance_map_v = cropping_center(distance_map_v, (80,80))

            distance_map = np.concatenate((distance_map_h.squeeze()[:,:,np.newaxis], distance_map_v.squeeze()[:,:,np.newaxis]), axis=-1)
        
        img = cv2.resize(img, (self.w, self.h))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        img = np.stack((img, img, img), axis=0)

        mask = mask
        distance_map = distance_map.transpose((2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.int64)).long()
        distance_map = torch.from_numpy(distance_map.astype(np.float32))

        class_mask = torch.stack((mask==0, mask!=0), dim = 0)

        return {'image' : img, 'label' : mask, 'distance_map' : distance_map, 'class_mask' : class_mask}
    

def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array
    
    Returns:
        x: cropped array
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x