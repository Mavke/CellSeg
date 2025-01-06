import numpy as np
import skimage
from scipy import ndimage

def get_distance(point, points, lower_scale = -1, upper_scale = 1):
    distances = (points - point)
    if np.sum(distances) == 0:
        return np.zeros(points.shape[0])
    
    normalized_distance = (upper_scale + 1) *(distances - np.min(distances)) / (np.max(distances) - np.min(distances)) + (lower_scale)


    return normalized_distance

def relable_masks(masks: np.ndarray):
    new_labeled = np.zeros(masks.shape).astype(np.int16)
    for i in range(masks.shape[0]):
        mask = masks[i,:,:,:]
        max_label = 0
        for cl in range(1,masks.shape[-1]):
            cl_mask = mask[:,:,cl]
            new_cl_mask = skimage.measure.label(cl_mask, background=0)
            tmp = np.max(new_cl_mask)
            new_cl_mask[np.where(new_cl_mask!= 0) ] += max_label
            max_label = tmp
            new_labeled[i,:,:,cl] = new_cl_mask
    return new_labeled

def get_center_of_masses(masks: np.ndarray, labels):
    com = np.zeros((labels.shape[0], 2))
    for i, label in enumerate(labels):
        x, y = np.where(masks == label)
        center_x = np.mean(x)
        center_y = np.mean(y)
        com[i,0] = center_x
        com[i,1] = center_y
    if np.isnan(com).any():
        print('nan values in center of mass')
    return com

def get_distance_maps(masks: np.ndarray):
    distance_map_h = np.zeros(masks.shape[:-1])
    distance_map_v = np.zeros(masks.shape[:-1])

    invalid_data = 0
    for i in range(masks.shape[0]):
        mask = masks[i,:,:,:].squeeze()
        tmp_distance_map_h = np.zeros(mask.shape[:-1])
        tmp_distance_map_v = np.zeros(mask.shape[:-1])
        mask_unique, c = np.unique(mask,return_counts=True)
        if np.sum(np.where(c > 1)) > 0:
            invalid_data += 1
        for cl in range(1, masks.shape[-1]):
            cl_mask = mask[:,:,cl]
            unique_labels, counts = np.unique(cl_mask, return_counts=True)

            unique_labels = unique_labels[1:]

                
            com = get_center_of_masses(cl_mask,unique_labels)
            com_np = np.asarray(com)
            for j,l in enumerate(unique_labels):
                if l == 0:
                    continue
                
                label_mask = (cl_mask == l)

                y_indices, x_indices = np.where(label_mask)
                center_x, center_y = com_np[j,:]


                normalized_h = get_distance(center_y, x_indices)
                normalized_v = get_distance(center_x, y_indices)
                if np.sum(normalized_h) == 0 or np.sum(normalized_v) == 0:
                    normalized_h = np.zeros(x_indices.shape)
                    normalized_v = np.zeros(y_indices.shape)
                tmp_distance_map_h[y_indices, x_indices] = normalized_h
                
                tmp_distance_map_v[y_indices, x_indices] = normalized_v
        
        distance_map_h[i] = tmp_distance_map_h
        distance_map_v[i] = tmp_distance_map_v
    return distance_map_h, distance_map_v
    
def get_distance_maps_single_mask(mask: np.ndarray):
    distance_map_h = np.zeros(mask.shape)
    distance_map_v = np.zeros(mask.shape)
    
    
    unique_labels, counts = np.unique(mask, return_counts=True)

    unique_labels = unique_labels[1:]

        
    com = get_center_of_masses(mask,unique_labels)
    com_np = np.asarray(com)
    for j,label in enumerate(unique_labels):
        if label == 0:
            continue
        
        label_mask = (mask == label)

        y_indices, x_indices = np.where(label_mask)
        center_x, center_y = com_np[j,:]


        normalized_h = get_distance(center_y, x_indices)
        normalized_v = get_distance(center_x, y_indices)
        
        distance_map_h[y_indices, x_indices] = normalized_h
        
        distance_map_v[y_indices, x_indices] = normalized_v

    return distance_map_h, distance_map_v

def get_overlapping_edges(instance_mask):
    shared_edges = np.zeros_like(instance_mask, dtype=np.int16)
    for i in np.unique(instance_mask):
        if i == 0:
            continue
        instance = instance_mask == i
        structuring_element = np.ones((3, 3))
        erosion_mask = ndimage.binary_erosion(instance, structuring_element)
        dilation_mask = ndimage.binary_dilation(instance, structuring_element)

        edge = np.asarray(dilation_mask, dtype=np.int16) - np.asarray(erosion_mask, dtype=np.int16)
        shared_edges += edge
    return shared_edges > 1