"""
threshold_utils.py

This Python module provides a collection of utility functions for performing statistical thresholding on numerical data. The primary focus is on determining threshold values that highlight significant data points based on predefined criteria. Additionally, the module supports resampling images and applying thresholding techniques for binary segmentation.

Functions

- `find_top_alpha_percent_threshold(array, alpha)`:
    - Computes the threshold value that retains the top `alpha%` of non-zero values in the given array. This function is useful for isolating the most significant data points.

- `resample_and_threshold_image(image, target_affine, target_shape, output_path=None)`:
    - Resamples the input image to the specified affine transformation and shape. After resampling, it applies Otsu's method to threshold the image, effectively segmenting it into binary regions. The resulting image can optionally be saved to the specified output path.
"""

import numpy as np
import nibabel as nib
from nilearn.image import resample_img
from skimage.filters import threshold_otsu 

def find_top_alpha_percent_threshold(array, alpha):
    """
    Computes the threshold value that retains the top alpha% of non-zero values in a given array. 

    Parameters:
    array (numpy.ndarray): The input array.
    alpha (float): The proportion of the array values to keep. Should be between 0 and 1.

    Returns:
    float: The threshold value to keep the top alpha% of the non-zero values.
    """
    # Check if alpha is between 0 and 1
    if not (0 < alpha < 1):
        raise ValueError("Alpha should be a float between 0 and 1.")

    # Flatten the array and filter out zeros
    flattened_array = array.flatten()
    non_zero_array = flattened_array[flattened_array != 0]

    # If there are no non-zero values, return 0
    if len(non_zero_array) == 0:
        return 0

    # Calculate the index for the top alpha%
    threshold_index = int(np.ceil((1 - alpha) * len(np.sort(non_zero_array))))

    return np.sort(non_zero_array)[threshold_index]


def resample_and_threshold_image(image, target_affine, target_shape, output_path=None):
    """
    Resample the image to the target affine and shape, then apply Otsu's method to threshold it.

    Parameters:
    image (Nifti1Image): Original image.
    target_affine (numpy.ndarray): Target affine matrix.
    target_shape (tuple): Target shape.
    output_path (Path, optional): Path to save the binary mask. Defaults to None.

    Returns:
    binary_mask_img (Nifti1Image): Resampled and binarized image.
    """
    resampled_img = resample_img(image, target_affine=target_affine, target_shape=target_shape)
    threshold = threshold_otsu(resampled_img.get_fdata())
    binary_mask_data = (resampled_img.get_fdata() >= threshold).astype(np.uint8)
    binary_mask_img = nib.Nifti1Image(binary_mask_data, resampled_img.affine)
    if output_path:
        nib.save(binary_mask_img, output_path)
    return binary_mask_img