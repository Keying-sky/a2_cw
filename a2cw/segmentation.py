import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_nii(data_path):
    """
    Load CT scans and segmentation masks for all patients.
    
    Param:
        data_path: Path object, path to the .nii files
    
    Returns:
        scans: Dictionary, keys are IDs (str),  values are scanned arrays
        masks: Dictionary, keys are IDs (str), values are mask arrays
    """
    scans = {}
    masks = {}
    
    files = os.listdir(data_path)
    
    ids = set()
    
    for file in files:            
        if file.endswith('.nii'):
            if '_mask' in file:
                id = file.split('_mask')[0].split('_')[1]  # case_id_mask.nii
            else:
                id = file.split('.nii')[0].split('_')[1]   # case_id.nii 
            ids.add(id)
    
    for id in ids:
        scan_file = next((f for f in files if f == f"case_{id}.nii"), None)
        mask_file = next((f for f in files if f == f"case_{id}_mask.nii"), None)
        
        if scan_file and mask_file:
            scan_path = os.path.join(data_path, scan_file)
            mask_path = os.path.join(data_path, mask_file)
            
            # load NIfTI file
            scan_nifti = nib.load(scan_path)
            mask_nifti = nib.load(mask_path)
            
            # convey to array
            scan_data = scan_nifti.get_fdata()
            mask_data = mask_nifti.get_fdata()
            
            scans[id] = scan_data
            masks[id] = mask_data
    
    return scans, masks

def seg_range(mask):
    """
    Find the range of non-zero values in the mask.
    
    Param: 
        mask: 3D array, mask for segmentation
        
    Return:
        ranges: Dictionary, containing the minimum and maximum in each direction.
    """
    # find the indexs of non-zero values
    idxs = np.where(mask > 0)
    
    if len(idxs[0]) == 0:
        return None
    
    ranges = {
        'x': (int(np.min(idxs[0])), int(np.max(idxs[0]))),
        'y': (int(np.min(idxs[1])), int(np.max(idxs[1]))),
        'z': (int(np.min(idxs[2])), int(np.max(idxs[2])))
    }
    
    return ranges

def subvolume(scan, range, pad=(30, 30, 5)):
    """
    Creating scanned subvolumes based on segmentation range.
    
    Params:
        scan: 3D array, the whole scan
        range: Dictionary, containing the minimum and maximum in each direction.
        pad: Tuple (x_pad, y_pad, z_pad)
    
    Return:
        subv: 3D array, scanned subvolume
    """
    if range is None:
        return None
    
    x_min, x_max = range['x']
    y_min, y_max = range['y']
    z_min, z_max = range['z']
    
    # padding
    x_min_pad = max(0, x_min - pad[0])
    x_max_pad = min(scan.shape[0], x_max + pad[0])
    
    y_min_pad = max(0, y_min - pad[1])
    y_max_pad = min(scan.shape[1], y_max + pad[1])
    
    z_min_pad = max(0, z_min - pad[2])
    z_max_pad = min(scan.shape[2], z_max + pad[2])
    
    subv = scan[x_min_pad:x_max_pad, y_min_pad:y_max_pad, z_min_pad:z_max_pad]
    
    return subv

def segment(subvolume, mask, threshold_min=None, threshold_max=None):
    """
    Threshold segmentation.
    
    Params:
        subvolume: 3D array, the subvolume scanned
        mask: 3D array, true segmentation mask
        threshold_min: Minimum threshold
        threshold_max: Maximum threshold
    
    Return: 3D array, segmentation result 
    """
    if threshold_min is None or threshold_max is None:
        voxel_values = subvolume[np.where(mask > 0)]
        
        if threshold_min is None:
            threshold_min = np.percentile(voxel_values, 10)
        
        if threshold_max is None:
            threshold_max = np.percentile(voxel_values, 90)
    
    segmentation = np.zeros_like(subvolume)
    segmentation[(subvolume >= threshold_min) & (subvolume <= threshold_max)] = 1
    
    return segmentation

def otsu_segment(subvolume):
    """
    Otsu thresholding segmentation.
    
    Params:
        subvolume: 3D array, the subvolume scanned
        mask: 3D array, true segmentation mask (not used in Otsu method)
        threshold_min: Minimum threshold (not used in Otsu method)
        threshold_max: Maximum threshold (not used in Otsu method)
    
    Return: 3D array, segmentation result 
    """
    from skimage.filters import threshold_otsu
    
    segmentation = np.zeros_like(subvolume)
    
    flat_data = subvolume.flatten()
    
    try:
        otsu_threshold = threshold_otsu(flat_data)
        
        segmentation[subvolume > otsu_threshold] = 1
    except Exception as e:
        # if fails (e.g., data change is small), fall back to the base threshold
        print(f"Otsu thresholding failed: {e}. Using basic thresholding instead.")
        segmentation[subvolume > np.mean(subvolume)] = 1  # use the mean as threshold
    
    return segmentation

def evaluate(segmentation, ground_truth):
    """
    Compare segmentation results with true labels
    
    Params:
        segmentation: 3D array, segmentation result
        ground_truth: 3D array, true segmentation mask
    
    Return:
        metrics: Dictionary of evaluation metrics
    """
    # ensure that segmentation and true labels are binary
    seg_binary = (segmentation > 0).astype(np.int32)
    gt_binary = (ground_truth > 0).astype(np.int32)
    
    # true positive, false positive, true negative and false negative
    TP = np.sum((seg_binary == 1) & (gt_binary == 1))
    FP = np.sum((seg_binary == 1) & (gt_binary == 0))
    TN = np.sum((seg_binary == 0) & (gt_binary == 0))
    FN = np.sum((seg_binary == 0) & (gt_binary == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'dice': dice,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    }
    
    return metrics

def visualise(scan, truth, segmentation, slice=None):
    """
    Visualise the raw scans, true segmentation and threshold segmentation.
    
    Params:
        scan: 3D array, raw scan or subvolume
        truth: 3D array, true segmentation mask
        segmentation: 3D array, segmentation result
        slice: The index of the displayed slice
    """
    if slice is None:
        slice = scan.shape[2] // 2    # default the middle slice

    scan_slice = scan[:, :, slice]
    tr_slice = truth[:, :, slice]
    seg_slice = segmentation[:, :, slice]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(scan_slice, cmap='gray')
    axes[0].set_title('Original Scan')
    axes[0].axis('off')
    
    axes[1].imshow(scan_slice, cmap='gray')
    axes[1].imshow(tr_slice, cmap='gray', alpha=0.5)
    axes[1].set_title('Truth Segmentation')
    axes[1].axis('off')
    
    axes[2].imshow(scan_slice, cmap='gray')
    axes[2].imshow(seg_slice, cmap='gray', alpha=0.5)
    axes[2].set_title('Threshold Segmentation')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
