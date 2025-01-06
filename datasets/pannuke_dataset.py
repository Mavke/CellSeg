import os
import numpy as np
import torch, copy, cv2
import albumentations as A


from zipfile import ZipFile
from torch.utils.data import Dataset
from datasets.data_set_prep import *
from post_processing.pannuke_pq import remap_label


class PannukeDataset(Dataset):
    """
        Dataset class for the pannuke dataset. Loads the whole dataset into ram to boost the loading performance.
    """
    def __init__(self, data_path: str, fold: int, output_size: list, split: str, ref_std = None, ref_mean = None, model_type = 'instance'):
        self.split = split
        self.model_type = model_type
        folder_name = 'Fold ' + str(fold)
        sub_folder = "fold" + str(fold)

        images_path = os.path.join(folder_name, 'images', sub_folder, 'images.npy')
        masks_path = os.path.join(folder_name, 'masks', sub_folder, 'masks.npy')
        type_path = os.path.join(folder_name, 'images', sub_folder, 'types.npy')

        self.transform = A.Compose([
            A.Rotate((90,90), interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomScale(scale_limit=[0.5, 0.5], p = 0.15),
            A.Blur(blur_limit=10, p=0.2),
            A.GaussNoise(p=0.25, var_limit=50),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.2),
            A.Superpixels(p_replace=0.1,n_segments=200, max_size=output_size[0]/2, p=0.1),
            A.ZoomBlur(max_factor=1.05, p=0.1),
            A.ElasticTransform(sigma=25, alpha=0.5, alpha_affine=15, p=0.2),
            # A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255.0)
        ])

        with ZipFile(data_path) as zip_file:
            self.images = np.load(zip_file.open(images_path)).astype(np.uint8)
            self.types = np.load(zip_file.open(type_path))
            self.masks = np.load(zip_file.open(masks_path)).astype(np.uint16)
            self.true_masks = copy.deepcopy(self.masks)

            indices = []

            # initialize arrays with (number ofImages, width, height)
            non_overlap_mask = np.zeros(self.masks.shape[:-1], dtype=np.int8)
            non_overlap_mask = (self.masks[:,:,:,0] != 0).astype(np.int8)
            
            #iterate over all classes and detect overlapping pixels
            for cl in range(1,self.masks.shape[-1]):
                cl_mask = (self.masks[:,:,:,cl].astype(np.int8) != 0).astype(np.int8)
                overlap = non_overlap_mask * (cl_mask != 0).astype(np.int8)
                non_overlap_mask *= (1-overlap).astype(np.int8)

                cl_mask = cl_mask * (1-overlap).astype(np.int8)
                non_overlap_mask += cl_mask
            self.masks *= non_overlap_mask[:,:,:,np.newaxis].astype(np.uint8)
            self.instance_mask = copy.deepcopy(self.masks)

            # remove class 5 (background) set value to 0
            for cl in range(0, self.masks.shape[-1]):
                self.masks[:,:,:,cl] = (self.masks[:,:,:,cl] != 0) * ((cl+1) % self.masks.shape[-1])
            self.masks[:,:,:,5] = 0

            
            #condense masks to one channel
            self.masks = np.sum(self.masks[...,:-1], axis=-1)
            valid_indices = []
            self.instance_mask = np.sum(self.instance_mask[...,:-1], axis=-1)

            #remove images without any mask
            for i in range(self.masks.shape[0]):
                if np.sum(self.masks[i]) != 0:
                    valid_indices.append(i)
            
            self.masks = self.masks[valid_indices]
            self.types = self.types[valid_indices]
            self.images = self.images[valid_indices]
            self.true_masks = self.true_masks[valid_indices]
            class_mask_copy = copy.deepcopy(self.true_masks)
        
            self.true_masks[:, :, :, 5] = class_mask_copy[:, :, :, 0]
            self.true_masks[:, :, :, 0] = class_mask_copy[:, :, :, 5]

            del class_mask_copy
            self.instance_mask = self.instance_mask[valid_indices]

        self.w, self.h = output_size


    def __len__(self):
        return self.images.shape[0]


    def getRefValues(self):
        return self.mean_ref, self.std_ref


    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx].astype(np.int8)
        type = self.types[idx]
        
        if self.model_type == 'semantic':
            
            instances = self.instance_mask[idx]
            edges = get_overlapping_edges(instances)
            mask = mask * (1-edges)

        if self.split == 'train':
            masks = [mask, self.true_masks[idx]]
            transformed = self.transform(image=image, masks=masks)
            image = transformed['image']
            mask = transformed['masks'][0]
            class_mask = transformed['masks'][1] 

            labeled_mask = remap_label(np.expand_dims(transformed['masks'][1],0))
            distance_map_h, distance_map_v = get_distance_maps(labeled_mask)
            distance_map = np.concatenate((distance_map_h.squeeze()[:,:,np.newaxis], distance_map_v.squeeze()[:,:,np.newaxis]), axis=-1)

           
        elif self.split != 'train':
            class_mask = self.true_masks[idx]
            labeled_mask = relable_masks(np.expand_dims(self.true_masks[idx],0))

            distance_map_h, distance_map_v = get_distance_maps(labeled_mask)
            distance_map = np.concatenate((distance_map_h.squeeze()[:,:,np.newaxis], distance_map_v.squeeze()[:,:,np.newaxis]), axis=-1)

        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        image = (image - np.min(image)) / (np.max(image) - np.min(image))

 
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        class_mask = cv2.resize(class_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        distance_map = cv2.resize(distance_map, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        mask = np.asarray(mask, np.uint8)

        #one hote encode class mask
        class_mask = class_mask != 0

        image = image.transpose((2, 0, 1))
        distance_map = distance_map.transpose((2, 0, 1))
        class_mask = class_mask.transpose((2, 0, 1))
        instance_mask = self.instance_mask[idx]

        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.uint8)).long()
        class_mask = torch.from_numpy(class_mask.astype(np.uint8)).long()
        distance_map = torch.from_numpy(distance_map.astype(np.float32))
        instance_mask = torch.from_numpy(instance_mask.astype(np.int16)).long()

        return {'image' : image, 'label' : mask, 'distance_map' : distance_map, 'class_mask' : class_mask, 'instance_mask' : instance_mask, 'type': type}
    
import numpy as np
from PIL import Image




def _normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:$
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    # HERef = np.array([[0.5626, 0.2159],
    #                   [0.7201, 0.8012],
    #                   [0.4062, 0.5581]])
        
    # maxCRef = np.array([1.9705, 1.0308])

    maxCRef= np.array([1.9713055649557338, 0.741354425035508])
    HERef = np.array(
    [
        [0.5001340654085598, 0.004804369872676684],
        [0.7272425313652708, 0.7330272758823506],
        [0.47008958421915664, 0.6801822776599128],
    ]
    )
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile+'.png')
        Image.fromarray(H).save(saveFile+'_H.png')
        Image.fromarray(E).save(saveFile+'_E.png')

    return Inorm, H, E
    