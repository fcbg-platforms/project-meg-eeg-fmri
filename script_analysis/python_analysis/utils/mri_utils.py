"""
mri_utils.py

This module provides utility function for fMRI data preprocessing: statistical thresholding of MRI data.

Functions:
----------
1. mri_data_thresholding(local_dir: str, subjects: list, tasks_contrasts: dict, logger: logging.Logger, plotting: bool = False):
    Applies statistical thresholding to fMRI data for a list of subjects and specified tasks/contrasts. 
    Optionally generates visualizations of the results. This function also handles the creation and 
    masking of brain images, and saves the processed outputs in a BIDS-compliant directory structure.
"""
import nibabel as nib
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_stat_map
from nilearn.image import resample_img
from nilearn.maskers import NiftiMasker

# Personal Imports
from utils.nifti_utils import extract_dof_from_niftiheader, extract_threshold
from utils.contrast_utils import p_value_to_zscore, process_t2zscore
from utils.utils import trackprogress
from utils.mask_utils import process_ribbon
from utils.threshold_utils import resample_and_threshold_image

from config import CorrectionParam

def mri_data_thresholding(local_dir: str, subjects: list, tasks_contrasts: dict, logger: logging.Logger, plotting: bool = False):
    """
    Performs thresholding of MRI data for a list of subjects and specified tasks/contrasts, with the option to visualize results.

    This function processes MRI data for multiple subjects, applies statistical thresholding on the z-score maps generated from t-contrast images, and saves the thresholded z-score maps in BIDS-compliant directories. Optionally, it can plot and display the results. The function handles the resampling of brain masks, masking of z-score maps, and thresholding of the resulting masked data. It also logs all the operations and handles any errors that may occur during the process.

    Parameters:
    -----------
    local_dir : str
        The base directory where the MRI data and derivative output directories are located.
        
    subjects : list
        A list of subject IDs (strings) to be processed. Each ID should correspond to a folder within the specified `local_dir`.

    tasks_contrasts : dict
        A dictionary where keys are task names (e.g., 'aud', 'vis') and values are lists of contrast names (e.g., 'contrast1', 'contrast2') to be processed for each task.

    logger : logging.Logger
        A logger object used to record events, warnings, errors, and progress during the processing. This allows for monitoring and debugging of the function's operations.

    plotting : bool, optional (default = False)
        If True, the function will generate and display plots of the thresholded z-score maps and brain masks.

    Returns:
    --------
    None
        The function does not return any values. It saves the processed and thresholded z-score maps in the appropriate directories within `local_dir`. Progress and issues encountered during the process are logged using the provided logger.

    Workflow:
    ---------
    1. **Initialization:** The function starts a timer to track the processing time and calculates the total number of subjects.
    2. **Subject Loop:** For each subject:
        - Loads the T1-weighted anatomical image.
        - Processes the ribbon mask to create a brain mask.
        - Optionally plots the brain mask.
        - Iterates through each task associated with the subject.
    3. **Task Loop:** For each task:
        - Initializes statistical correction parameters such as alpha level, correction method, and cluster size.
        - Iterates through each contrast associated with the task.
    4. **Contrast Loop:** For each contrast:
        - Checks if the z-score map already exists; if not, it computes the z-score map from the t-contrast image.
        - Resamples the brain mask to match the z-score image's resolution.
        - Applies the brain mask to the z-score image.
        - Saves the masked z-score map.
        - Performs statistical thresholding on the masked z-score map.
        - Optionally plots the thresholded z-score map.
        - Adds metadata to the NIfTI header and saves the thresholded z-score map.
    """

    # Start the time count to track the remaining computation time
    start_time = time.time()
    
    # Total number of subjects for progress tracking
    tot_perf = len(subjects)  
    
    # Loop through subjects
    for count_perf, sub in enumerate(subjects):
        try:
            print(f"Analyzing {sub}.")
            logger.info(f"Analyzing {sub}.")
    
            # Track progress
            trackprogress(count_perf, tot_perf, start_time, logger)

            sub_derivatives_dir = Path(local_dir) / 'derivatives' / sub

            # Load the T1 image if plotting
            if plotting == True:
                anat_path = Path(local_dir) / 'bids' / sub / 'anat' 
                T1_path = anat_path / f"{sub}_T1w.nii.gz"
                T1_img = nib.load(T1_path)
                logger.info(f"Loaded T1 image for {sub}.")
    
            # Def derivatives anat outdir
            brainmask_outdir = sub_derivatives_dir / 'anat'
            brainmask_outdir.mkdir(parents=True, exist_ok=True)
    
            # From the ribbon mask, create a brain_mask that will be used to mask the fMRI data
            brain_mask = process_ribbon(sub, local_dir, brainmask_outdir)
            if plotting == True:
                # Plot the brain mask
                plot_stat_map(brain_mask, T1_img, title=f'Brain mask {sub}')
                plt.show()
                plt.close()
    
            # Loop through tasks
            for task, contrasts in tasks_contrasts.items():
                try:
                    logger.info(f"  Analyzing the {task} task.")
                    print(f"  Analyzing the {task} task.")
    
                    # Initialize correction parameters for the current task
                    correc_param = CorrectionParam(task)
                    alpha = correc_param.fMRI_ALPHA
                    corr_method = correc_param.fMRI_CORRECTION_METHOD
                    clustersize = correc_param.fMRI_CLUSTER_THRES
                    twosided = correc_param.fMRI_TWOSIDED
                    
                    # Define the output directory for contrasts 
                    contrast_outdir = sub_derivatives_dir / 'func' / f'task-{task}'
                    contrast_outdir.mkdir(parents=True, exist_ok=True)
    
                    # Loop through contrasts
                    for index, contrast in enumerate(contrasts):
                        logger.info(f"    Running contrast {contrast}")
                        print(f"    Running contrast {contrast}")
                        
                        try:
                            # File path for the z-score map
                            zscore_path = contrast_outdir / f"{sub}_task-{task}_contrast-{contrast}_desc-stat-z_statmap.nii.gz"
                            
                            # If the z-score map does not exist, compute and save it, else load it
                            if not zscore_path.exists():
                                # Load the tcontrast
                                t_contrast_path = sub_derivatives_dir / 'func' / f'task-{task}' / f"{sub}_task-{task}_contrast-{contrast}_desc-stat-t_statmap.nii.gz"
                                t_contrast = nib.load(t_contrast_path)
                                
                                # Get the degree of freedom from the NIfTI header
                                dof = extract_dof_from_niftiheader(t_contrast.header)
    
                                # Transform to zscore
                                z_nifti = process_t2zscore(t_contrast, dof, index + 1, contrast, zscore_path)
    
                                # Free space
                                del t_contrast
    
                            else:
                                z_nifti = nib.load(zscore_path)
                                dof = extract_dof_from_niftiheader(z_nifti.header)
    
                            # Resample the brain mask to fit to the fMRI data 
                            resample_brain_mask_path = brainmask_outdir / f"{sub}_brainmask_desc-resamp2fmri.nii.gz"
                            if not resample_brain_mask_path.exists():
                                resample_brain_mask = resample_and_threshold_image(brain_mask, z_nifti.affine, z_nifti.shape, resample_brain_mask_path)
                            else:
                                resample_brain_mask = nib.load(resample_brain_mask_path)
    
                            # Mask the zscore
                            brain_mask_nifti_masker = NiftiMasker(resample_brain_mask)
                            masked_zscore = brain_mask_nifti_masker.inverse_transform(brain_mask_nifti_masker.fit_transform(z_nifti))
    
                            # Save it
                            masked_zscore_path = Path(f"{zscore_path.with_suffix('').with_suffix('')}_masked.nii.gz")
                            nib.save(masked_zscore, masked_zscore_path)
                            logger.info(f"      Masked z-score saved at {masked_zscore_path}")
    
                            # Proceed to thresholding the z-score map
                            threshold_zscore, thres = threshold_stats_img(masked_zscore, alpha=alpha, height_control=corr_method,
                                                                          cluster_threshold=clustersize, two_sided=twosided)
                            if plotting == True:
                                plot_stat_map(threshold_zscore, T1_img, threshold=thres, display_mode="mosaic", cmap=plt.cm.hot,
                                              title=f"{corr_method}, alpha = {alpha}, n > {clustersize}, corrected, Task = {task}, Contrasts = {contrast}, {sub}")
                                plt.show()
                                plt.close()
                                
                            # Define the output directory and file path for the thresholded z-score map
                            thres_outdir = contrast_outdir / f'corr-{corr_method}'
                            thres_outdir.mkdir(parents=True, exist_ok=True)
    
                            # Define the path for the thresholded z-score map
                            thresholded_path = thres_outdir / f"{masked_zscore_path.with_suffix('').with_suffix('').name}_corr-{corr_method}_alpha-{alpha}_cluster-{clustersize}_twosided-{twosided}.nii.gz"
    
                            # Add information in the NIfTI header description
                            description = f"{{Z_[{dof}]}} - contrast {index + 1}: {contrast}, threshold = {round(thres, 3)}, alpha = {alpha}, n >  {clustersize}, {corr_method}".encode('utf-8')[:80]
                            new_descrip = np.array(description, dtype='|S80')
                            threshold_zscore.header['descrip'] = new_descrip
    
                            # Save the thresholded z-score map
                            nib.save(threshold_zscore, thresholded_path)
                            logger.info(f"      Thresholded z-score saved at {thresholded_path}")
                            
                        except Exception as e:
                            logger.error(f"    Error processing contrast {contrast}: {str(e)}")
                            # Remove all handlers associated with the root logger
                            raise e
                except Exception as e:
                    logger.error(f"  Error processing task {task}: {str(e)}")
                    raise e
        except Exception as e:
            logger.error(f"Error processing subject {sub}: {str(e)}")
            raise e
        
    logger.info("Processing completed.")
    print("Processing completed.")