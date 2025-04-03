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

def boundary_aware(subvolume, init_seg, boundary):
    """
    An adaptive boundary segmentation method.
    
    Params:
        subvolume: subvolume of the CT scaned
        init_seg: Initial threshold segmentation
        boundary: Widths of outer boundary (3D)
    
    Return:
        refined_seg: Result after boundary segmentation.
    """
    import numpy as np
    from scipy import ndimage
    
    x_size, y_size, z_size = subvolume.shape
    bound_mask = np.zeros_like(subvolume, dtype=bool)
    
    bound_mask[:boundary[0], :, :] = True
    bound_mask[x_size-boundary[0]:, :, :] = True
    bound_mask[:, :boundary[1], :] = True
    bound_mask[:, y_size-boundary[1]:, :] = True
    bound_mask[:, :, :boundary[2]] = True
    bound_mask[:, :, z_size-boundary[2]:] = True
    
    # mark every segments
    labeled_segments, n_seg = ndimage.label(init_seg)

    refined_seg = np.zeros_like(init_seg)

    for segment_id in range(1, n_seg + 1):
        current_seg = (labeled_segments == segment_id)
        # split current segment to A (outside boundary) and B (inside boundary)
        A = current_seg & bound_mask
        B = current_seg & ~bound_mask
        
        # calculate sizes
        A_voxels = np.sum(A)
        B_voxels = np.sum(B)
        
        # completely outside the boundary, remove all
        if B_voxels == 0:
            continue  
        
        # completely inside the boundary，save all
        if A_voxels == 0:
            refined_seg = refined_seg | current_seg
            continue
        
        # for fragments riding on the boundary
        if A_voxels > B_voxels:
            # find the largest other internal fragements
            other_internal_segments = np.zeros_like(init_seg, dtype=bool)
            for other_id in range(1, n_seg + 1):
                if other_id != segment_id:
                    other_segment = (labeled_segments == other_id)
                    other_internal = other_segment & ~bound_mask
                    other_internal_segments = other_internal_segments | other_internal
            
            # find the largest internal fragment (if has)
            if np.any(other_internal_segments):
                labeled_other, n_other = ndimage.label(other_internal_segments)
                if n_other > 0:
                    other_sizes = np.bincount(labeled_other.ravel())
                    other_sizes[0] = 0  # ignore the background
                    max_other_size = np.max(other_sizes) if len(other_sizes) > 1 else 0
                    
                    # remove A, save B
                    if B_voxels > max_other_size:
                        refined_seg = refined_seg | B
                    else:
                        # remove (A+B)
                        continue
                else:
                    # save B
                    refined_seg = refined_seg | B
            else:
                # save B
                refined_seg = refined_seg | B
        else:
            # save A+B
            refined_seg = refined_seg | current_seg
    
    return refined_seg

def adapt_thd_segm(subv, mask=None, margin_factor=0.1, boundary=None, morph=True):         
    """
    An adaptive threshold segmentation method with adaptive boundary aware segmentation.
    
    Params:
        subv: subvolume of the CT scaned
        mask: Reference mask to calculate the initial threshold
        margin_factor: Threshold min/max expansion factor
        boundary: Widths of outer boundary (3D)
        morph: Whether to apply morphology clean up
    
    Return:
        seg: the final segmentation
    """
    import numpy as np
    from scipy import ndimage
    
    if mask is not None:
        mask_indices = np.where(mask > 0)
        voxel_values = subv[mask_indices]
        
        thrh_min = np.percentile(voxel_values, 10)
        thrh_max = np.percentile(voxel_values, 90)
        
        range_width = thrh_max - thrh_min
        thrh_min -= range_width * margin_factor
        thrh_max += range_width * margin_factor
    else:
        # otsu method if no reference mask
        from skimage.filters import threshold_otsu
        thrh_otsu = threshold_otsu(subv)
        
        thrh_min = thrh_otsu
        thrh_max = np.max(subv)

        seg = np.zeros_like(subv, dtype=np.uint8)
        seg[(subv >= thrh_min) & (subv <= thrh_max)] = 1
        return seg

    # initial segmentation
    init_seg = np.zeros_like(subv, dtype=np.uint8)
    init_seg[(subv >= thrh_min) & (subv <= thrh_max)] = 1
    
    if morph:
        init_seg = ndimage.binary_opening(init_seg, structure=np.ones((2,2,2)))
        init_seg = ndimage.binary_closing(init_seg, structure=np.ones((3,3,3)))
    
    if boundary is not None:
        seg = boundary_aware(subv, init_seg, boundary)
        if morph:
            seg = ndimage.binary_opening(seg, structure=np.ones((1,1,1)))
    else:
        seg = ndimage.binary_opening(init_seg, structure=np.ones((1,1,1)))
    
    return seg

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
