import numpy as np
from skimage.transform import radon, iradon, iradon_sart, resize 
from scipy.ndimage import zoom
## ----------------------------- Ex1.1 ----------------------------- ##
def correct_ct(sino, dark, flat):
    """
    Correct the CT sinogram using dark and flat fields.
    
    Params:
        sino : ndarray, raw CT sinogram data
        dark : ndarray, dark field measurement (background noise)   
        flat : ndarray, flat field measurement (reference beam without sample)

    Return:
        ndarray, corrected sinogram 
    """
    sino_corrected = sino - dark
    
    denominator = flat - dark
    denominator[denominator <= 0] = 1.0 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        corrected = -np.log(sino_corrected / denominator)
    
    # infinity and NaN
    corrected[np.isinf(corrected) | np.isnan(corrected)] = 0
    
    return corrected

def correct_pet(sino, calibration):
    """
    Correct the PET sino using detector gain calibration.
    
    Params:
        sino : ndarray, raw PET sino data
        calibration : ndarray, detector gain calibration data

    Return:
        ndarray, corrected sino  
    """
    calibration_safe = np.copy(calibration)
    calibration_safe[calibration_safe == 0] = 1.0
    
    corrected = sino / calibration_safe
    
    return corrected

## ----------------------------- Ex1.2 ----------------------------- ##
def fbp(sino, theta):
    """
    Reconstruct CT image using FBP method.
    
    Params:
        sino: Corrected CT sino
        theta: Projection Angle Array
        
    Return: 
        Reconstructed CT image
    """

    recons = iradon(sino, theta=theta, filter_name='ramp', interpolation='linear', circle=True)
                    
    return recons

def os_sart(sino, theta, n_iter=50, n_subset=10, relaxation=0.5):
    """
    Reconstruct CT image using OS-SART method.
    
    Params:
        sino : Corrected CT sinogram
        theta: Projection Angle Array
        n_iter: Number of iterations
        relaxation: The relaxation parameter
        num_subset: Number of subsets
        
    Return: 
        Reconstructed CT image
    """
    _, n_angles = sino.shape
    
    image_shape = iradon(sino[:, :1], theta=[theta[0]], filter_name=None).shape
    x = np.zeros(image_shape)
    
    angle_idxs = np.arange(n_angles)
    subsets = np.array_split(angle_idxs, n_subset)
    
    for i in range(n_iter):
        for _, subset in enumerate(subsets):
            subset_theta = theta[subset]
            subset_sino = sino[:, subset]

            x = iradon_sart(
                subset_sino, 
                theta=subset_theta,
                image=x,
                relaxation=relaxation
            )

            x[x < 0] = 0

    return x

def sirt(sino, theta, n_iter=20, relaxation=0.5):
    """
    Reconstruct CT image using SIRT method.
    
    Params:
        sino : Corrected CT sinogram
        theta: Projection Angle Array
        n_iter: Number of iterations
        relaxation: The relaxation parameter
        
    Return: Reconstructed CT image
    """
    image_shape = iradon(sino[:, :1], theta=[theta[0]], filter_name=None).shape
    x = np.zeros(image_shape)
    
    for i in range(n_iter):
        x = iradon_sart(
            sino, 
            theta=theta,
            image=x,
            relaxation=relaxation
        )

        x[x < 0] = 0
         
    return x

## ----------------------------- Ex1.3 ----------------------------- ##
def resize_ct(ct_image, ct_psize=1.06, pet_psize=4.24):
    """
    Resize CT images to match the pixel size of PET images.
    
    Params:
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
    Convert a CT image to a 511 keV photon attenuation coefficient map.
    
    This function uses a piecewise linear transformation to convert Hounsfield Units (HU)
    to attenuation coefficients (mu) at 511 keV, which is the energy of PET photons. 
    
    Params:
        ct_image: Input CT image in Hounsfield Units
        
    Return: Attenuation coefficient map, representing the linear attenuation coefficients at 511 keV
        """
    attenuation_map = np.zeros_like(ct_image)

    # Air and Lungs (HU < -950)
    air_mask = ct_image < -950
    attenuation_map[air_mask] = 0.0
    
    # soft tissue (-950 <= HU <= 50)
    soft_tissue_mask = (-950 <= ct_image) & (ct_image <= 50)
    attenuation_map[soft_tissue_mask] = 9.6e-5 * (ct_image[soft_tissue_mask] + 1000)
    
    # skeleton (HU > 50)
    bone_mask = ct_image > 50
    attenuation_map[bone_mask] = 3.64e-5 * (ct_image[bone_mask] + 1000) + 0.0626
    
    return attenuation_map

def attenuation_sino(attenuation_map, pet_sino_shape):
    """
    Generate a sinogram of attenuation coefficients from an attenuation map.

    Params:
        attenuation_map: 2D array of attenuation coefficients at 511 keV  
        pet_sino_shape: Shape of the target PET sinogram
            
    Return:
        Sinogram of attenuation values
    """
    n_angle = pet_sino_shape[1]
    
    theta = np.linspace(0, 180, n_angle, endpoint=False)
    
    attenuation_sino = radon(attenuation_map, theta=theta, circle=True)

    if attenuation_sino.shape[1] != pet_sino_shape[1]:
        scale_factor = pet_sino_shape[1] / attenuation_sino.shape[1]
        attenuation_sino = zoom(attenuation_sino, 
                               (1, scale_factor),
                               order=1) 
    
    return attenuation_sino

def attenuation_corr(pet_sino, attenuation_sino):
    """
    Apply attenuation correction to a PET sinogram.
    
    Params:
        pet_sino: Uncorrected PET sinogram data
        attenuation_sino: Sinogram of attenuation values
        
    Return:
        Attenuation-corrected PET sinogram
    """
    acf = np.exp(attenuation_sino)

    pet_sino_attn_corrected = pet_sino * acf
    
    return pet_sino_attn_corrected

## ----------------------------- Ex1.4 ----------------------------- ##
def osem(sino, theta=None, n_iter=10, n_subset=10):
    """
    OSEM method to reconstruct the PET image.

    Params:
        sino: PET sinogram to be reconstructed
        theta: Projection Angle Array
        n_iter: Number of iterations
        n_subset: Number of splited subsets

    Returns:
        x: Reconstruction pet image
        intermediates: The intermediate images for visulisation
    """
    if theta is None:
        n_angle = sino.shape[1]
        theta = np.linspace(0, 180, n_angle, endpoint=False)
    
    _, n_angle = sino.shape
    
    image_shape = iradon(sino[:, :1], theta=[theta[0]], filter_name=None).shape
    x = np.ones(image_shape)
    
    indices = np.arange(n_angle)
    np.random.shuffle(indices)
    subsets = np.array_split(indices, n_subset)
    
    # store intermediate results to visualise the convergence process
    intermediates = []
    
    for iter in range(n_iter):
        if iter % 2 == 0: 
            intermediates.append(x.copy())
        
        for _, subset in enumerate(subsets):
            subset_theta = theta[subset]
            subset_sino = sino[:, subset]
            
            forward_projection = radon(x, theta=subset_theta, circle=True)
            
            forward_projection[forward_projection < 1e-8] = 1e-8
            
            ratio = subset_sino / forward_projection
            
            backprojection = iradon(ratio, theta=subset_theta, filter_name=None, circle=True)              
            
            ones_backprojection = iradon(np.ones_like(ratio), theta=subset_theta, filter_name=None, circle=True)
                                       
            ones_backprojection[ones_backprojection < 1e-8] = 1e-8
            
            x *= backprojection / ones_backprojection

            x[x < 0] = 0

    intermediates.append(x.copy())
    
    return x, intermediates

def mlem(sino, theta=None, n_iter=10):
    """
    MLEM method to reconstruct the PET image.

    Params:
        sino: PET sinogram to be reconstructed
        theta: Projection Angle Array
        n_iter: Number of iterations

    Returns:
        x: Reconstruction pet image
        intermediates: The intermediate images for visulisation
    """
    if theta is None:
        n_angle = sino.shape[1]
        theta = np.linspace(0, 180, n_angle, endpoint=False)
    
    _, n_angle = sino.shape
    
    # initialise
    image_shape = iradon(sino[:, :1], theta=[theta[0]], filter_name=None).shape
    x = np.ones(image_shape)
    
    # store intermediate results to visualise the convergence process
    intermediates = []
    
    for iter in range(n_iter):
        if iter % 2 == 0: 
            intermediates.append(x.copy())
        
        forward_projection = radon(x, theta=theta, circle=True)
        
        forward_projection[forward_projection < 1e-8] = 1e-8

        ratio = sino / forward_projection
        
        backprojection = iradon(ratio, theta=theta, filter_name=None, circle=True)

        ones_backprojection = iradon(np.ones_like(ratio), theta=theta, filter_name=None, circle=True)
                                   
        ones_backprojection[ones_backprojection < 1e-8] = 1e-8

        x *= backprojection / ones_backprojection
        
        x[x < 0] = 0
    
    intermediates.append(x.copy())
    
    return x, intermediates