"""
contrast_utils.py

This module provides utility functions for processing and converting t-fMRI contrast data into z-scores, facilitating statistical analysis of brain imaging data. The primary focus is on transforming t-statistic NIfTI images into z-score maps, which are more interpretable for assessing statistical significance. 

Functions:
- process_t2zscore(fmri_contrast, dof, index=None, con_corresp=None, zscore_path=None, dofmax=DEF_DOFMAX):
    Converts t-statistic fMRI contrast data into z-scores, updating the NIfTI header with relevant metadata. 
    Optionally saves the z-score NIfTI image to a specified path.

- p_value_to_zscore(pvalue, one_minus_pvalue=None):
    Converts p-values into z-scores, providing a standardized measure of statistical significance.
"""


import numpy as np
from scipy.stats import norm, t
import nibabel as nib

DEF_DOFMAX = 1e10

def process_t2zscore(fmri_contrast, dof, index = None, con_corresp = None, zscore_path = None, dofmax = DEF_DOFMAX):
    """
    Convert t-statistic fMRI contrast data to z-scores and optionally save the resulting NIfTI image.

    Parameters
    ----------
    fmri_contrast : nib.Nifti1Image
        The t-fMRI contrast NIfTI image.
    dof : float
        The degree of freedom.
    index : int (default = None)
        The contrast index.
    con_corresp : str (default = None)
        The contrast correspondence description.
    zscore_path : str (default = None)
        The file path where the new z-score NIfTI image will be saved.

    dofmax : float (default = DEF_DOFMAX)
        The maximum degree of freedom.

    Returns
    -------
    z_nifti: nib.Nifti1Image
        The Z-score fMRI contrast NIfTI image.
    None
    """
    # Extract the t-statistic data
    t_data = fmri_contrast.get_fdata()
    
    # Compute p-values and one_minus_pvalues
    p_values = t.sf(t_data, np.minimum(dof, dofmax))
    one_minus_pvalues = t.cdf(t_data, np.minimum(dof, dofmax))
    
    # Convert p-values to z-scores
    z_score = p_value_to_zscore(p_values, one_minus_pvalue=one_minus_pvalues)

    new_header = fmri_contrast.header.copy()
    
    # Create a new NIfTI header with the updated description
    if index != None and con_corresp != None:
        description = f"{{Z_[{dof}]}} - contrast {index}: {con_corresp}".encode('utf-8')[:80]
        new_descrip = np.array(description, dtype='|S80')
        new_header['descrip'] = new_descrip
    
    # Create and save the new NIfTI image
    z_nifti = nib.Nifti1Image(z_score, fmri_contrast.affine, new_header)

    if zscore_path != None:
        nib.save(z_nifti, zscore_path)
    
    # print(f"Saved new z-score NIfTI file to {zscore_path}")
    return(z_nifti)

    # Extract the t-statistic data
    t_data = fmri_contrast.get_fdata()

    # Compute p-values and one_minus_pvalues
    p_values = t.sf(t_data, np.minimum(dof, dofmax))
    one_minus_pvalues = t.cdf(t_data, np.minimum(dof, dofmax))

    # Convert p-values to z-scores
    z_score = p_value_to_zscore(p_values, one_minus_pvalue=one_minus_pvalues)
    logger.debug("Converted p-values to z-scores.")


    # Create a new NIfTI header with the updated description
    new_header = fmri_contrast.header.copy()
    if index is not None and con_corresp is not None:
        description = f"{{Z_[{dof}]}} - contrast {index}: {con_corresp}".encode('utf-8')[:80]
        new_descrip = np.array(description, dtype='|S80')
        new_header['descrip'] = new_descrip
        logger.info(f"Updated NIfTI header description: {description.decode('utf-8')}")

    # Create and save the new NIfTI image
    z_nifti = nib.Nifti1Image(z_score, fmri_contrast.affine, new_header)
    logger.info("Created new NIfTI image with z-scores.")

    if zscore_path is not None:
        nib.save(z_nifti, zscore_path)
        logger.info(f"Saved new z-score NIfTI file to {zscore_path}")

    return z_nifti

def p_value_to_zscore(pvalue, one_minus_pvalue=None):
    """Return the z-score(s) corresponding to certain p-value(s) and, \
    optionally, one_minus_pvalue(s) provided as inputs.

    Parameters
    ----------
    pvalue : np.ndarray
        An array of p-values computed using the survival function. The shape 
        of this array should match the shape of the t-contrast.
    
    one_minus_pvalue : np.ndarray, optional
        An array of (1 - p-value) values, which can be computed using the 
        cumulative distribution function. This array should also match the 
        shape of the t-contrast. If provided, it should be the value returned 
        by `nilearn.glm.contrasts.one_minus_pvalue`. The number of elements in 
        `one_minus_pvalue` should be equal to the number of elements in 
        `pvalue`.
    
    Returns
    -------
    z_scores : np.ndarray
        An array of z-scores with the same shape as the input t-contrast.

    Notes
    -------
    Function inspired by nilearn.glm._utils.z_score function from nilearn

    """
    # Clip p-values to avoid extreme values that could cause numerical issues
    pvalue = np.clip(pvalue, 1.0e-300, 1.0 - 1.0e-16)
    
    # Compute z-scores from p-values using the inverse survival function (ISF)
    z_scores_sf = norm.isf(pvalue)
    
    if one_minus_pvalue is not None:
        # Clip 1 - p-values to avoid extreme values that could cause numerical issues
        one_minus_pvalue = np.clip(one_minus_pvalue, 1.0e-300, 1.0 - 1.0e-16)
        
        # Compute z-scores from 1 - p-values using the percent-point function (PPF)
        z_scores_cdf = norm.ppf(one_minus_pvalue)
        
        # Prepare an empty array to store z-scores
        z_scores = np.empty(pvalue.shape)
        
        # Determine which p-values use the survival function (SF) and which use the CDF
        use_cdf = z_scores_sf < 0
        use_sf = np.logical_not(use_cdf)
        
        # Assign z-scores based on which function (CDF or SF) is used
        z_scores[np.atleast_1d(use_cdf)] = z_scores_cdf[use_cdf]
        z_scores[np.atleast_1d(use_sf)] = z_scores_sf[use_sf]
    else:
        # If one_minus_pvalue is not provided, use only the survival function results
        z_scores = z_scores_sf

    return z_scores