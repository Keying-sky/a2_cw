import numpy as np

def butterworth_lowpass_filter(shape, D0=30, n=2):
    P, Q = shape[0], shape[1]
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / D0) ** (2 * n))
    return H

## ------------------- Evaluate denoising quality ------------------- ##
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def eval_denois(origi_im, denoised_im, noise_region=None, signal_region=None):
    """
    Evaluate the performance of image denoising by comparing original and denoised images.
    
    Params:
        origi_im: Original image before denoising    
        denoised_im: Denoised image to be evaluated           
        noise_region: Boolean mask indicating regions of noise for SNR calculation   
        signal_region: Boolean mask indicating regions of signal for SNR calculation
           
    Return:
        Dictionary containing calculated metrics
    """
    results = {}

    orig_norm = (origi_im - np.min(origi_im)) / (np.max(origi_im) - np.min(origi_im))
    den_norm = (denoised_im - np.min(denoised_im)) / (np.max(denoised_im) - np.min(denoised_im))
    
    results['MSE'] = mse(orig_norm, den_norm)  # MSE
    
    results['PSNR'] = psnr(orig_norm, den_norm)  # PSNR
    
    results['SSIM'] = ssim(orig_norm, den_norm, data_range=1.0)  # SSIM
    
    # SNR
    if signal_region is not None and noise_region is not None:
        orig_signal = np.mean(origi_im[signal_region])
        orig_noise = np.std(origi_im[noise_region])
        results['Original_SNR'] = orig_signal / orig_noise if orig_noise > 0 else float('inf')
        
        den_signal = np.mean(denoised_im[signal_region])
        den_noise = np.std(denoised_im[noise_region])
        results['Denoised_SNR'] = den_signal / den_noise if den_noise > 0 else float('inf')
        
        results['SNR_Improvement'] = results['Denoised_SNR'] / results['Original_SNR'] if results['Original_SNR'] > 0 else float('inf')
    return results

def compare_methods(origi_im, denoised_ims_dict, noise_region=None, signal_region=None):
    """
    Compare multiple denoising methods using various image quality metrics.
    
    Params
    ----------
    origi_im: Original image before denoising   
    denoised_ims_dict: Dictionary mapping method names to their corresponding denoised images   
    noise_region: Boolean mask indicating regions of noise for SNR calculation
    signal_region: Boolean mask indicating regions of signal for SNR calculation
        
    Return: Nested dictionary with method names as keys and their evaluation results as values.
            Each value is a dictionary containing the metrics from eval_denois()
    """
    all_results = {}

    for method_name, denoised_im in denoised_ims_dict.items():
        all_results[method_name] = eval_denois(
            origi_im, denoised_im, noise_region, signal_region
        )
    
    # create comparison charts
    metrics = ['PSNR', 'SSIM']
    methods = list(denoised_ims_dict.keys())
    
    for metric in metrics:
        if all(metric in all_results[method] for method in methods):
            plt.figure(figsize=(10, 6))
            values = [all_results[method][metric] for method in methods]
            
            plt.bar(methods, values, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
            plt.title(f'{metric} Comparison')
            plt.ylabel(metric)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            for i, v in enumerate(values):
                plt.text(i, v + 0.01 * max(values), f"{v:.4f}", ha='center')
            
            plt.tight_layout()
            plt.show()
    
    # SNR comparison
    if all('Denoised_SNR' in all_results[method] for method in methods):
        plt.figure(figsize=(10, 6))
        values = [all_results[method]['Denoised_SNR'] for method in methods]
        
        plt.bar(methods, values, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
        plt.axhline(y=all_results[methods[0]]['Original_SNR'], color='r', linestyle='--', 
                   label=f'Original SNR ({all_results[methods[0]]["Original_SNR"]:.2f})')
        plt.title('SNR Comparison Across Methods')
        plt.ylabel('SNR Value')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(values):
            plt.text(i, v + 0.01 * max(values), f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.show()
    
    return all_results

def s_n_masks(magnitude_image, signal_threshold=0.2, noise_threshold=0.1):
    """
    Generate signal and noise masks for an image based on intensity thresholds.
    
    Params:
        magnitude_image: Input image from which to generate masks   
        signal_threshold: Normalised intensity threshold to identify signal regions
        noise_threshold: Normalised intensity threshold to identify noise regions           
        
    Returns:
        signal_mask : Boolean array indicating signal regions
        noise_mask : Boolean array indicating noise regions
    """
    # normalise
    img_norm = (magnitude_image - np.min(magnitude_image)) / (np.max(magnitude_image) - np.min(magnitude_image))
    
    signal_mask = img_norm > signal_threshold
    noise_mask = img_norm < noise_threshold
    
    # morphological operations to smooth masks
    signal_mask = ndimage.binary_opening(signal_mask, structure=np.ones((3,3)))
    signal_mask = ndimage.binary_closing(signal_mask, structure=np.ones((5,5)))
    
    noise_mask = ndimage.binary_opening(noise_mask, structure=np.ones((3,3)))
    noise_mask = ndimage.binary_closing(noise_mask, structure=np.ones((3,3)))
    
    # ensure the noise region doesn't contain the signal region
    noise_mask = noise_mask & ~signal_mask
    
    return signal_mask, noise_mask

def visual_masks(image, signal_mask, noise_mask):
    """
    Visualise the original image along with signal and noise masks.
    
    Parameters
        image: Original image to visualise     
        signal_mask: Boolean mask indicating signal regions (displayed in green)      
        noise_mask: Boolean mask indicating noise regions (displayed in red)      
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image, alpha=0.9, cmap='gray')
    plt.imshow(signal_mask, alpha=0.3, cmap='Greens')
    plt.title('Signal Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image, alpha=0.9, cmap='gray')
    plt.imshow(noise_mask, alpha=0.5, cmap='Reds')
    plt.title('Noise Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()