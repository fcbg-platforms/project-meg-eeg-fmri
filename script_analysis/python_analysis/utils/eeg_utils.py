"""
This module is designed for processing and analyzing EEG data, specifically focusing on the conversion of EEG source estimates into NIFTI format and their interpolation into voxel space.

Functions:
- eeg_data_SEstim2nifti(path_to_data, logger, subjects, tasks_conditions, task_tp,  thres_param, plotting): Converts EEG source estimate files to NIFTI format and thresholds them for further analysis. It is the main function of this module.
- interpolate_sestim_to_voxel_space(nifti_img, coordinates, values): Interpolates source 
  estimate values to the voxel space of a given NIfTI image.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import numpy as np
import glob
import time
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from nilearn.plotting import plot_stat_map
from nilearn.image import resample_img
from nilearn.glm import threshold_stats_img
from nilearn.maskers import NiftiMasker

from pycartool.spi import read_spi 
from pycartool.ris import read_ris

# Personal import
from utils.utils import trackprogress
from utils.window_utils import get_max_index, get_poa_timepoint_sestim
from utils.mask_utils import process_ribbon 
from utils.threshold_utils import find_top_alpha_percent_threshold, resample_and_threshold_image

# Main function
def eeg_data_stc2nifti(path_to_data, logger, subjects, tasks_conditions, task_tp,  thres_param, stc_algo = 'LAURA', plotting = False):
    """
    Converts EEG source estimate files to NIFTI format and thresholds them for further analysis.

    This function performs the following tasks:
    1. **Data Preparation**: Converts EEG source estimate files into NIFTI format through interpolation, aligning them with the anatomical brain images. This enables comparison with fMRI data.
    2. **Thresholding and Masking**: Applies a user-defined threshold based on the top percentile alpha level and a minimum cluster size criterion to the NIFTI images. The data are also resampled to match the fMRI resolution.

   Parameters:
    - `path_to_data`: `str`  
      The base directory where the subject data is stored.
    - `logger`: `logging.Logger`  
      A logging object used to record the progress of processing and any issues encountered.
    - `subjects`: `list of str`  
      A list of subject identifiers for whom the data will be processed.
    - `tasks_conditions`: `dict`  
      A dictionary where each key is a task and each value is a list of conditions associated with that task.
    - `task_tp`: `dict`  
     A dictionary where each key represents a task, and the corresponding value is either a list of time point windows of interest or,
     for conditions, another dictionary where each key is a condition and each value is a list of time point windows associated with that condition.
    - `thres_param`: `dict`  
      A dictionary containing thresholding parameters:
        - `alpha`: `float`  
        The top percentile for thresholding.
      - `cluster_thresh`: `int`  
        The minimum cluster size for the thresholding process.
    - `stc_algo`: str
        The algorithm used to compute the source estimates. Default is 'LAURA'.
    - plotting : bool, optional
        If True, generates and displays plots of the thresholded images. Default is False.

    The function tracks progress and estimates remaining time for processing each subject, and saves the processed images and visualizations to disk.
    """

    # Start the time count to track the remaining computation time
    start_time = time.time()
    
    # Total number of subjects for progress tracking
    tot_perf = len(subjects) 
    
    # Threshold percentile
    logger.info(f'Selected threshold top percentile alpha: {thres_param['EEG']['alpha']}')
    
    # Minimum cluster size
    logger.info(f'Selected cluster size > {thres_param['EEG']['cluster_thresh']}')
    
    # Loop through subjects
    for count_perf, sub in enumerate(subjects):
        try:
            logger.info(f'\nProcessing {sub}')
            print(f'\nProcessing {sub}')
    
            trackprogress(count_perf, tot_perf, start_time, logger)
                
            sub_derivatives_dir = Path(path_to_data) / 'derivatives' / sub 
        
            # Define the path to the ESI for this subject
            path_sub_data = sub_derivatives_dir / 'eeg-esi' 
            path_source_space = path_sub_data / "T1.spi"
            # Load the source space
            logger.info(f'  Loading source space from: {path_source_space}')
            try:
                source_space = read_spi(path_source_space)
                # Path to the anatomical image
                anat_path = path_sub_data / "T1.nii"
                logger.info(f'  Loading anatomical image from: {anat_path}')
                try:
                    anat_img = nib.load(anat_path)
                except Exception as e:
                    logger.error(f'  Failed to load the anatomical image from: {anat_path}. Error: {e}')
                    print(f'  Failed to load the anatomical image from: {anat_path}. Error: {e}')
                    raise
                    
            except Exception as e:
                logger.warning(f'  Failed to load source space from: {path_source_space}. Error: {e}')
                path_source_space = path_sub_data / "T1.BFC.spi"
                logger.info(f'  Loading source space from: {path_source_space}')
                try:
                    source_space = read_spi(path_source_space)
                    # Path to the anatomical image
                    anat_path = path_sub_data / "T1.BFC.nii"
                    logger.info(f'  Loading anatomical image from: {anat_path}')
                    try:
                        anat_img = nib.load(anat_path)
                    except Exception as e:
                        logger.error(f'  Failed to load the anatomical image from: {anat_path}. Error: {e}')
                        raise
                except Exception as e:
                    logger.error(f'  Failed to load source space from: {path_source_space}. Error: {e}')
                    raise
        
            coordinates_sspace = source_space.coordinates
        
            # Define directories and paths
            brainmask_outdir = sub_derivatives_dir / 'anat'
            brainmask_path = brainmask_outdir / f"{sub}_brainmask_desc-resamp2fmri.nii.gz"
        
            if not brainmask_path.exists():
                logger.info(f'  Brain mask not found, creating a new one.')
                # Create a brain mask from the ribbon mask if it doesn't exist
                brain_mask = process_ribbon(sub, path_to_data, brainmask_outdir)
                
                # Load fMRI data to obtain the target affine and shape for resampling
                fmri_data_outdir =  sub_derivatives_dir / 'func' / 'task-aud'
                path_fmri_data = fmri_data_outdir / f"{sub}_task-aud_contrast-sine_desc-stat-t_statmap.nii.gz"
                logger.info(f'  Loading fMRI data from: {path_fmri_data}')
                fmri_data_img = nib.load(path_fmri_data)
            
                logger.info(f'  Resampling brain mask to match fMRI resolution.')
                resample_brain_mask = resample_and_threshold_image(
                    brain_mask, fmri_data_img.affine, fmri_data_img.shape[:3], brainmask_path
                )
                # Free memory
                del brain_mask, fmri_data_outdir, path_fmri_data, fmri_data_img
            else:
                logger.info(f'  Loading existing resampled brain mask from: {brainmask_path}')
                resample_brain_mask = nib.load(brainmask_path)
            
               
            # Loop through tasks
            for task in tasks_conditions.keys():
                print(f"  \nAnalysis of task {task}")
                logger.info(f"  \nAnalysis of task {task}")
                try:
                    outdir =  sub_derivatives_dir / 'eeg' / 'stc_interpolated' / f'task-{task}'
                    outdir.mkdir(parents=True, exist_ok=True)
                    
                    # Loop through conditions
                    for condition in tasks_conditions[task]:
                        logger.info(f'    Processing condition: {condition}')
                        print(f'    Processing condition: {condition}')
                        try:
                            pattern = path_sub_data / f"{sub}_task-{task}_proc-raw_EEG_*_*_{condition}.icacorr.SpatialFilter.{stc_algo} * RGV.ZScoreVN.Vectorial Mean *.ris"
                            matching_files = list(glob.glob(str(pattern)))
                    
                            if len(matching_files) == 1: 
                                esi_data_path = matching_files[0]
                            elif len(matching_files) > 1:
                                raise ValueError(f'Multiple matching files found, it should be only one: {matching_files}.')
                            else:
                                raise ValueError(f'No file found matching the pattern {pattern}.')
                    
                            logger.info(f'      Loading source estimate from: {esi_data_path}')
                            source_estimate = read_ris(esi_data_path)
                    
                            # Found the windows for this condition
                            if isinstance(task_tp[task], dict):
                                windows = task_tp[task][condition]
                            else:
                                windows = task_tp[task]
                                
                            # Loop through the windows of interest    
                            for window in windows:
                                try:
                                    logger.info(f'      Processing window: {window}')
                                    values_window = np.linalg.norm(source_estimate.sources_tc[:, :, window[0]:window[1]], axis=1)
                                    sestim_tp_values, tp = get_poa_timepoint_sestim(values_window, window)
                        
                                    logger.info(f'        Interpolating to anatomical space at time point: {tp}')
                                    interpolated_data = interpolate_sestim_to_voxel_space(anat_img, coordinates_sspace, sestim_tp_values)
                                    interpolated_img = nib.Nifti1Image(interpolated_data, anat_img.affine, anat_img.header)
                                    
                                    interpolated_path = outdir / f"{sub}_task-{task}_condition-{condition}_desc-eeg-stcinterpol_tp-{tp}_stat-{stc_algo}_statmap.nii.gz"
                                    nib.save(interpolated_img, interpolated_path)
                                    logger.info(f'        Saved interpolated image to: {interpolated_path}')
                        
                                    logger.info(f'        Resampling the interpolated image.')
                                    interpolated_img_resamp = resample_img(interpolated_img, resample_brain_mask.affine, resample_brain_mask.shape, interpolation = 'linear')
                                    interpolated_path_resamp = outdir / f"{interpolated_path.with_suffix('').stem}_resamp.nii.gz"
                                    nib.save(interpolated_img_resamp, interpolated_path_resamp)
                                    
                                    logger.info(f'        Masking the resampled image.')
                                    brain_mask_nifti_masker = NiftiMasker(mask_img=resample_brain_mask)
                                    interpolated_img_masked = brain_mask_nifti_masker.inverse_transform(
                                        brain_mask_nifti_masker.fit_transform(interpolated_img_resamp)
                                    )
                                    interpolated_path_masked = outdir /  f"{interpolated_path_resamp.with_suffix('').stem}_masked.nii.gz"
                                    nib.save(interpolated_img_masked, interpolated_path_masked)
                        
                                    logger.info(f'        Thresholding at the top alpha percentile.')
                                    threshold = find_top_alpha_percent_threshold(interpolated_img_masked.get_fdata(), thres_param['EEG']['alpha'])
                                    interpolated_img_thres, _ = threshold_stats_img(interpolated_img_masked, threshold = threshold, cluster_threshold = thres_param['EEG']['cluster_thresh'], height_control = None)
                                    if plotting == True:
                                        # Plot the thresholded image
                                        plot_stat_map(
                                            interpolated_img_thres, anat_img, threshold = threshold, display_mode="mosaic",
                                            cmap=plt.cm.hot, title=f"Interpolated EEG source estimate, task = {task}, condition = {condition}, tp = {tp}, {sub}"
                                        )
                                        plt.show()
                                        plt.close()
                        
                                    # Save the thresholded image
                                    interpolated_path_thres = outdir / f"{interpolated_path_masked.with_suffix('').stem}_topercent-{thres_param['EEG']['alpha']}_cluster-{thres_param['EEG']['cluster_thresh']}.nii.gz"
                    
                                    # Add a description the the image header with the threshold value
                                    description = f"{condition}, threshold = {round(threshold, 3)}, alpha = {thres_param['EEG']['alpha']}, n > {thres_param['EEG']['cluster_thresh']}".encode('utf-8')[:80]
                                    interpolated_img_thres.header['descrip'] = description
                                    nib.save(interpolated_img_thres, interpolated_path_thres)
                                    logger.info(f'        Saved thresholded image to: {interpolated_path_thres}')
                                    
                                except Exception as e:
                                    logger.error(f"      Error processing window {window}: {str(e)}")
                                    raise e
             
                        except Exception as e:
                            logger.error(f"    Error processing condition {condition}: {str(e)}")
                            raise e
                except Exception as e:
                    logger.error(f"  Error processing task {task}: {str(e)}")
                    raise e
        except Exception as e:
            logger.error(f"Error processing subject {sub}: {str(e)}")
            raise e
        
    logger.info("Processing completed.")
    

def interpolate_sestim_to_voxel_space(nifti_img, coordinates, values):
    """
    Interpolates source estimate values to the voxel space of a given NIfTI image.

    Parameters:
    nifti_img (nibabel.Nifti1Image): The NIfTI image defining the voxel space for interpolation.
    coordinates (numpy.ndarray): The coordinates (in world space) where the source estimates are located.
    values (numpy.ndarray): The source estimate values corresponding to the provided coordinates.

    Returns:
    numpy.ndarray: An array of interpolated source estimates in the voxel space defined by `nifti_img`.
    """
    # Get information about the NIfTI image
    affine = nifti_img.affine
    data = nifti_img.get_fdata()
    dims = data.shape

    # Create a grid for the voxel space
    grid_x, grid_y, grid_z = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]

    # Convert voxel grid to world coordinates
    voxel_coords = np.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T
    world_coords = nib.affines.apply_affine(affine, voxel_coords)

    # Interpolate the source estimates to the voxel grid
    interpolated_component = griddata(coordinates, values, world_coords, method='linear', fill_value=0.)

    # Reshape the interpolated component to match the voxel space dimensions
    interpolated_values = interpolated_component.reshape(dims)

    return interpolated_values


