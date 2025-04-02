import numpy as np
from skimage.transform import radon, iradon, iradon_sart, resize 
from scipy.ndimage import zoom
## ----------------------------- Ex1.1 ----------------------------- ##
def correct_ct(sinogram, dark, flat):
    """
    Correct the CT sinogram using dark and flat fields.
    
    Parameters:
        sinogram : ndarray, raw CT sinogram data
            
        dark : ndarray, dark field measurement (background noise)
            
        flat : ndarray, flat field measurement (reference beam without sample)

    Returns:
        ndarray, corrected sinogram 
    """
    sinogram_corrected = sinogram - dark
    
    denominator = flat - dark
    denominator[denominator <= 0] = 1.0 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        corrected = -np.log(sinogram_corrected / denominator)
    
    # infinity and NaN
    corrected[np.isinf(corrected) | np.isnan(corrected)] = 0
    
    return corrected

def correct_pet(sinogram, calibration):
    """
    Correct the PET sinogram using detector gain calibration.
    
    Parameters:
        sinogram : ndarray, raw PET sinogram data
        calibration : ndarray, detector gain calibration data

    Returns:
        ndarray, corrected sinogram  
    """
    calibration_safe = np.copy(calibration)
    calibration_safe[calibration_safe == 0] = 1.0
    
    corrected = sinogram / calibration_safe
    
    return corrected

## ----------------------------- Ex1.2 ----------------------------- ##
def fbp(sinogram, theta):
    """
    Reconstruct CT image using FBP method.
    
    Parameters:
        sinogram: Corrected CT sinogram
        theta: Projection Angle Array
        
    Returns: 
        Reconstructed CT image
    """

    recons = iradon(sinogram, theta=theta, filter_name='ramp', interpolation='linear', circle=True)
                    
    return recons

def os_sart(sinogram, theta, n_iter=50, n_subset=10, relaxation=0.5):
    """
    Reconstruct CT image using OS-SART method.
    
    Parameters:
        sinogram : Corrected CT sinogram
        theta: Projection Angle Array
        n_iter: Number of iterations
        relaxation: The relaxation parameter
        num_subset: Number of subsets
        
    Returns: 
        Reconstructed CT image
    """
    _, n_angles = sinogram.shape
    
    image_shape = iradon(sinogram[:, :1], theta=[theta[0]], filter_name=None).shape
    x = np.zeros(image_shape)
    
    angle_idxs = np.arange(n_angles)
    subsets = np.array_split(angle_idxs, n_subset)
    
    for i in range(n_iter):
        for subset_idx, subset in enumerate(subsets):
            subset_theta = theta[subset]
            subset_sino = sinogram[:, subset]

            x = iradon_sart(
                subset_sino, 
                theta=subset_theta,
                image=x,
                relaxation=relaxation
            )

            x[x < 0] = 0
            
        if (i + 1) % 5 == 0:
            print(f"OS-SART complete the iteration {i + 1}/{n_iter}")

    return x

def sirt(sinogram, theta, n_iter=20, relaxation=0.5):
    """
    Reconstruct CT image using SIRT method.
    
    Parameters:
        sinogram : Corrected CT sinogram
        theta: Projection Angle Array
        n_iter: Number of iterations
        relaxation: The relaxation parameter
        
    Returns: Reconstructed CT image
    """
    image_shape = iradon(sinogram[:, :1], theta=[theta[0]], filter_name=None).shape
    x = np.zeros(image_shape)
    
    for i in range(n_iter):
        x = iradon_sart(
            sinogram, 
            theta=theta,
            image=x,
            relaxation=relaxation
        )

        x[x < 0] = 0

        if (i + 1) % 5 == 0:
            print(f"SIRT complete the iteration {i + 1}/{n_iter}")
    
    return x

## ----------------------------- Ex1.3 ----------------------------- ##
def resize_ct(ct_image, ct_psize=1.06, pet_psize=4.24):
    """
    Resize CT images to match the pixel size of PET images.
    
    Parameters:
        ct_image: Reconstructed CT image
        ct_psize: CT pixel size (mm)
        pet_psize: PET pixel size (mm)
        
    Returns: 
        Resized CT image for PET size
    """
    # scale factor
    scale = ct_psize / pet_psize
    
    new_shape = np.round(np.array(ct_image.shape) * scale).astype(int)
    
    resized_image = resize(ct_image, new_shape, 
                          order=3,
                          mode='constant', 
                          anti_aliasing=True,
                          preserve_range=True)
    
    return resized_image

def attenuation_map(ct_image):
    """

    """
    attenuation_map = np.zeros_like(ct_image)

    # Air and Lungs (HU < -950)
    air_mask = ct_image < -950
    attenuation_map[air_mask] = 0.0
    
    # soft tissue (-950 <= HU <= 100)
    soft_tissue_mask = (-950 <= ct_image) & (ct_image <= 100)
    attenuation_map[soft_tissue_mask] = 0.096 * (1 + ct_image[soft_tissue_mask]/1000)
    
    # skeleton (HU > 100)
    bone_mask = ct_image > 100
    attenuation_map[bone_mask] = 0.096 * (1.42 + 0.00789 * (ct_image[bone_mask]/1000))
    
    return attenuation_map

def attenuation_sino(attenuation_map, pet_sino_shape):
    """
 
    """
    num_angles = pet_sino_shape[1]
    
    theta = np.linspace(0, 180, num_angles, endpoint=False)
    
    attenuation_sino = radon(attenuation_map, theta=theta, circle=True)

    if attenuation_sino.shape[1] != pet_sino_shape[1]:
        scale_factor = pet_sino_shape[1] / attenuation_sino.shape[1]
        attenuation_sino = zoom(attenuation_sino, 
                               (1, scale_factor),
                               order=1) 
    
    return attenuation_sino

def attenuation_corr(pet_sino, attenuation_sino):
    """

    """
    attenuation_correction_factors = np.exp(attenuation_sino)

    pet_sino_attn_corrected = pet_sino * attenuation_correction_factors
    
    return pet_sino_attn_corrected

## ----------------------------- Ex1.4 ----------------------------- ##
def osem(sinogram, theta=None, num_iterations=10, num_subsets=10):
    """

    """
    if theta is None:
        num_angles = sinogram.shape[1]
        theta = np.linspace(0, 180, num_angles, endpoint=False)
    
    _, num_angles = sinogram.shape
    
    image_shape = iradon(sinogram[:, :1], theta=[theta[0]], filter_name=None).shape
    x = np.ones(image_shape)
    
    indices = np.arange(num_angles)
    np.random.shuffle(indices)
    subsets = np.array_split(indices, num_subsets)
    
    # store intermediate results to visualise the convergence process
    intermediate_images = []
    
    for iteration in range(num_iterations):
        if iteration % 2 == 0: 
            intermediate_images.append(x.copy())
        
        for _, subset in enumerate(subsets):
            subset_theta = theta[subset]
            subset_sino = sinogram[:, subset]
            
            forward_projection = radon(x, theta=subset_theta, circle=True)
            
            forward_projection[forward_projection < 1e-8] = 1e-8
            
            ratio = subset_sino / forward_projection
            
            backprojection = iradon(ratio, theta=subset_theta, 
                                   filter_name=None, circle=True)
            
            ones_backprojection = iradon(np.ones_like(ratio), theta=subset_theta, 
                                       filter_name=None, circle=True)
            ones_backprojection[ones_backprojection < 1e-8] = 1e-8
            
            x *= backprojection / ones_backprojection

            x[x < 0] = 0

    intermediate_images.append(x.copy())
    
    return x, intermediate_images

def mlem(sinogram, theta=None, num_iterations=20):
    """

    """
    if theta is None:
        num_angles = sinogram.shape[1]
        theta = np.linspace(0, 180, num_angles, endpoint=False)
    
    _, num_angles = sinogram.shape
    
    # initialise
    image_shape = iradon(sinogram[:, :1], theta=[theta[0]], filter_name=None).shape
    x = np.ones(image_shape)
    
    # store intermediate results to visualise the convergence process
    intermediate_images = []
    
    for iteration in range(num_iterations):
        if iteration % 2 == 0: 
            intermediate_images.append(x.copy())
        
        forward_projection = radon(x, theta=theta, circle=True)
        
        forward_projection[forward_projection < 1e-8] = 1e-8

        ratio = sinogram / forward_projection
        
        backprojection = iradon(ratio, theta=theta, 
                               filter_name=None, circle=True)
        
        ones_backprojection = iradon(np.ones_like(ratio), theta=theta, 
                                   filter_name=None, circle=True)
        ones_backprojection[ones_backprojection < 1e-8] = 1e-8

        x *= backprojection / ones_backprojection
        
        x[x < 0] = 0
    
    intermediate_images.append(x.copy())
    
    return x, intermediate_images