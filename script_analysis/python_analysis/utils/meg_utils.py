"""
meg_utils.py

This module contains functions designed for processing MEG (Magnetoencephalography) data and converting it into formats compatible with neuroimaging tools. Specifically, it includes a function for converting MEG source estimates into NIfTI format, applying thresholding, and optionally masking the resulting images for further analysis or comparison with fMRI data.

Functions
-----------
meg_data_stc2nifti(path_to_data, logger, subjects, tasks_conditions, tasks_tp,  thres_param, pos, stc_algo, mri_resolution = False, do_brain_masking = False, plotting = False)
    Converts MEG source estimate files to NIFTI format, applies thresholding, and optionally masks and plots the data, for all subjects, tasks and conditions.
    
stc_tp_as_volume(stc, src, tp, dest = 'mri', mri_resolution=False, format="nifti1", fname = None)
    Export the source estimate at a specific time point as a NIfTI volume.

"""
import warnings
import nibabel as nib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

from mne import read_forward_solution, read_source_estimate, Info, SourceSpaces, convert_forward_solution, VolSourceEstimate

from nilearn.image import resample_img
from nilearn.glm import threshold_stats_img
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map

# Personal imports
from utils.utils import trackprogress
from utils.mask_utils import process_ribbon
from utils.window_utils import get_max_index, get_poa_timepoint_sestim, compute_window_indices
from utils.threshold_utils import find_top_alpha_percent_threshold, resample_and_threshold_image


# Main function
def meg_data_stc2nifti(path_to_data, logger, subjects, tasks_conditions, tasks_tp,  thres_param, pos, stc_algo, mri_resolution = False, do_brain_masking = False, plotting = False):
    """
    Converts MEG source estimate files to NIFTI format, applies thresholding, and optionally masks and plots the data, for all subjects, tasks and conditions.

    This function performs the following tasks:
    1. **Data Preparation**: Converts MEG source estimate files into NIFTI format by interpolating the source estimates to align with anatomical brain images. This conversion facilitates comparison with fMRI data.
    2. **Thresholding and Masking**: Applies a user-defined threshold based on the top percentile alpha level and a minimum cluster size criterion to the NIFTI images. The images can also be resampled to match the fMRI resolution and optionally masked to exclude non-brain regions.
    3. **Plotting**: Optionally generates visualizations of the thresholded data for review.

    Parameters
    ----------
    path_to_data : str
        The base directory where the subject data is stored, including the raw MEG data, anatomical images, and derivatives.
    logger : logging.Logger
        A logging object used to record progress, warnings, and errors encountered during processing.
    subjects : list of str
        A list of subject identifiers whose data will be processed. Each identifier should match the directory structure in `path_to_data`.
    tasks_conditions : dict
        A dictionary where each key is a task (e.g., 'task1', 'task2') and each value is a list of conditions associated with that task (e.g., ['condition1', 'condition2']).
    tasks_tp : dict
        A dictionary where each key represents a task, and the corresponding value is either a list of time point windows of interest or, for conditions, another dictionary where each key is a condition and each value is a list of time point windows associated with that condition.
    thres_param : dict
        A dictionary containing thresholding parameters:
        - 'alpha' : float
            The top percentile for thresholding. Values above this percentile will be considered significant.
        - 'cluster_thresh' : int
            The minimum cluster size (in voxels) required to retain a cluster after thresholding.
    pos : int
        Resolution of the source estimate to read
    stc_algo: str
        Algorithm used to obtain the source estimates
    mri_resolution : bool or tuple of float, optional
        If True, the output images are resampled to match the resolution of the MRI image during interpolation. Default is False.
    do_brain_masking : bool, optional
        If True, a brain mask is applied to exclude non-brain regions from the NIFTI interpolated images. Default is False.
    plotting : bool, optional
        If True, generates and displays plots of the thresholded images. Default is False.

    Returns
    -------
    None
        The function processes and saves the data, but does not return any value.

    Notes
    -----
    - The function handles data for multiple subjects, tasks, and conditions, iterating through each to process and analyze the MEG source estimates.
    - Progress and errors are logged to the provided logger, and progress tracking is performed to estimate remaining processing time.
    - Resampling and masking are performed as specified, with images saved to disk in the specified format and location.
    - Thresholding is applied based on the alpha percentile and minimum cluster size parameters, and thresholded images are saved with descriptive headers.
    - If `plotting` is enabled, visualizations of the thresholded images are generated for each subject, task, and condition.
    """
    # Start the time count to track the remaining computation time
    start_time = time.time()
    
    # Total number of subjects for progress tracking
    tot_perf = len(subjects) 
    
    # Threshold percentile
    logger.info(f'Selected threshold top percentile alpha: {thres_param['MEG']['alpha']}')
    
    # Minimum cluster size
    logger.info(f'Selected cluster size > {thres_param['MEG']['cluster_thresh']}')
    
    # Loop through subjects
    for count_perf, sub in enumerate(subjects):
    
        print(f"Beginning of processing, {sub}:")
        logger.info(f"Beginning of processing, {sub}:")
    
        trackprogress(count_perf, tot_perf, start_time, logger)

        sub_derivatives_dir = Path(path_to_data) / 'derivatives' / sub 
    
        try:
            if plotting == True:
                # Load the subjects T1
                anat_dir = Path(path_to_data) / 'bids' / sub / 'anat' 
                anat_path = anat_dir / f"{sub}_T1w.nii.gz"
                anat_img = nib.load(anat_path)
                logger.info(f"Loaded T1 image for {sub}.")
        
            # Define directories and paths and load brain mask
            brainmask_outdir = sub_derivatives_dir / 'anat' 
            brainmask_path = brainmask_outdir / f"{sub}_brainmask_desc-resamp2fmri.nii.gz"
            
            if not brainmask_path.exists():
                logger.info(f'  Brain mask not found, creating a new one.')
                # Create a brain mask from the ribbon mask if it doesn't exist
                brain_mask = process_ribbon(sub, path_to_data, brainmask_outdir, anat_img)
                
                # Load fMRI data to obtain the target affine and shape for resampling
                fmri_data_outdir = sub_derivatives_dir / 'func' / 'task-aud' 
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
    
            # Define forward and stc directory
            meg_dir = sub_derivatives_dir / 'meg' 
    
            # Load fwd data
            fwd_file = meg_dir / 'forward' / f"sub_{sub[-2:]}-pos_{pos}-fwd.h5"
            fwd = read_forward_solution(fwd_file)
            fwd["info"] = Info(fwd["info"])
            fwd["src"] = SourceSpaces(fwd["src"])
            fwd = convert_forward_solution(
                fwd, force_fixed=False, surf_ori=True, copy=False
            )
            
            # Iterate over tasks and their time windows
            for task, windows in tasks_tp.items():
                print(f"  Processing task: {task}")
                logger.info(f"  Processing task: {task}")
        
                try:
                    outdir = meg_dir / 'stc_interpolated' / f'task-{task}' 
                    outdir.mkdir(parents=True, exist_ok=True)
        
                    # Loop through conditions
                    for condition in tasks_conditions[task]:
                        print(f"    Processing condition: {condition}")
                        logger.info(f"    Processing condition: {condition}")
                        
                        try:
                            # Load the source estimates
                            stc_file = meg_dir / 'stc' / f"sub_{sub[-2:]}_task_{task}_proc_raw_meg-epo-{condition}-pos_{pos}-stc.h5"
                            stc = read_source_estimate(stc_file)
                
                            # Define start, step, and end times based on stc
                            t_start, t_step, t_end = stc.tmin, stc.tstep, stc.tmin + stc.tstep * (stc.shape[1] - 1)
    
                            for window in windows:
                                logger.info(f"      Time window: {window}s")
    
                                try:
                                    start_idx, end_idx = compute_window_indices(window, t_start, t_end, t_step, stc.shape[1])
                                    
                                    logger.info(f"      Indices: start={start_idx}, end={end_idx}")
                                    
                                    # Get the tp which corresponds to the time course maximum in this window
                                    _, tp = get_poa_timepoint_sestim(stc.data[:, start_idx:end_idx], [start_idx, end_idx])

                                    logger.info(f'      Interpolating to mri space at time point: {tp}')
                                    interpolated_path = outdir / f"{sub}_task-{task}_condition-{condition}_desc-meg-stcinterpol_tp-{tp}_pos-{pos}_stat-{stc_algo}_statmap.nii.gz"
                                    interpolated_img = stc_tp_as_volume(stc, fwd['src'], tp, dest = 'mri', mri_resolution=mri_resolution, format="nifti1", fname = interpolated_path)
                                    logger.info(f'      Saved interpolated image to: {interpolated_path}')
                        
                                    logger.info(f'        Resampling the interpolated image.')
                                    interpolated_img_resamp = resample_img(interpolated_img, resample_brain_mask.affine, resample_brain_mask.shape, interpolation = 'linear')
                                    interpolated_path_resamp = outdir / f"{interpolated_path.with_suffix('').stem}_resamp.nii.gz"
                                    nib.save(interpolated_img_resamp, interpolated_path_resamp)
                                    
                                    # Apply brain masking if enabled
                                    if do_brain_masking:
                                        logger.info(f'        Masking the resampled image.')
                                        brain_mask_nifti_masker = NiftiMasker(mask_img=resample_brain_mask)
                                        interpolated_img_masked = brain_mask_nifti_masker.inverse_transform(brain_mask_nifti_masker.fit_transform(interpolated_img_resamp))
                                        interpolated_path_masked = outdir / f"{interpolated_path_resamp.with_suffix('').stem}_masked.nii.gz"
                                        nib.save(interpolated_img_masked, interpolated_path_masked)
                                        
                                        # Update references to the masked image
                                        interpolated_img_ = interpolated_img_masked
                                        interpolate_img_path_ = interpolated_path_masked
                                    else:
                                        interpolated_img_ = interpolated_img_resamp
                                        interpolate_img_path_ = interpolated_path_resamp
                                        
                                    logger.info(f'        Thresholding at the top alpha percentile.')
                                    threshold = find_top_alpha_percent_threshold(interpolated_img_.get_fdata(), thres_param['MEG']['alpha'])
                                    interpolated_img_thres, _ = threshold_stats_img(interpolated_img_, threshold = threshold, cluster_threshold = thres_param['MEG']['cluster_thresh'], height_control = None)

                                    if plotting == True:
                                        # Plot the thresholded image
                                        plot_stat_map(
                                            interpolated_img_thres, anat_img, threshold = threshold, display_mode="mosaic",
                                            cmap=plt.cm.hot, title=f"Interpolated MEG source estimate, task = {task}, condition = {condition}, tp = {tp}, {sub}"
                                        )
                                        plt.show()
                                        plt.close()
                        
                                    # Save the thresholded image
                                    interpolated_path_thres = outdir / f"{interpolate_img_path_.with_suffix('').stem}_topercent-{thres_param['MEG']['alpha']}_cluster-{thres_param['MEG']['cluster_thresh']}.nii.gz"
                        
                                    # Add a description the the image header with the threshold value
                                    description = f"{condition}, threshold = {round(threshold, 3)}, alpha = {thres_param['MEG']['alpha']}, n > {thres_param['MEG']['cluster_thresh']}".encode('utf-8')[:80]
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




def stc_tp_as_volume(stc, src, tp, dest = 'mri', mri_resolution=False, format="nifti1", fname = None):
    """
    Export the source estimate at a specific time point as a NIfTI volume.

    This function converts the source estimate data for a given time point into a NIfTI volume,
    which can be saved and used for further analysis or visualization.

    Parameters
    ----------
    stc : VolSourceEstimate
        The volumetric source estimate containing the source data. This must be an instance of 
        `VolSourceEstimate` and should have data corresponding to the desired time points.
    src : SourceSpaces
        The source space used for the forward model. It should be of type 'volume' or 'mixed', 
        where 'mixed' includes both surface and volume elements.
    tp : int
        The index of the time point in `stc` data to be exported as a volume.
    dest : {'mri', 'surf'}, optional
        Defines the coordinate system for the output volume. If 'mri', the volume is aligned with 
        the original MRI coordinate system. If 'surf', the volume is aligned with the FreeSurfer 
        surface coordinate system (Surface RAS). Default is 'mri'.
    mri_resolution : bool or tuple of float, optional
        If True, the output volume is resampled to match the MRI resolution. If a tuple of three floats 
        is provided, it specifies the desired voxel size in millimeters (e.g., (2.0, 2.0, 2.0)). 
        Default is False. Note that resampling to MRI resolution can produce large files.
    format : {'nifti1', 'nifti2'}, optional
        The format of the NIfTI file. Can be 'nifti1' (default) or 'nifti2'.
    fname : str, optional
        The filename where the NIfTI volume should be saved (e.g., 'output.nii.gz'). If None, the 
        volume is not saved to a file.

    Returns
    -------
    img : Nifti1Image
        The NIfTI image object representing the volume at the specified time point.

    Notes
    -----
    This function is inspired by the `_interpolate_data` function from MNE's morph module. It handles 
    the conversion of volumetric source estimates to NIfTI format, including optional resampling to 
    match MRI resolution or other specified voxel sizes.
    """
    from mne.morph import _check_dep, _triage_output, _get_src_data, _check_subject_src
    from mne.utils import BunchConst, _validate_type
    from mne.source_estimate import _BaseVolSourceEstimate
    
    _check_dep(nibabel="2.1.0", dipy=False)
    NiftiImage, NiftiHeader = _triage_output(format)
    _validate_type(stc, _BaseVolSourceEstimate, "stc", "volume source estimate")
    assert src.kind in ("volume", "mixed")

    voxel_size_defined = False

    if isinstance(mri_resolution, (int, float)) and not isinstance(mri_resolution, bool):
        # Convert to tuple if given as a single numeric value
        mri_resolution = (float(mri_resolution),) * 3

    if isinstance(mri_resolution, tuple):
        _check_dep(nibabel=False, dipy="0.10.1")  # Ensure DIPY is available
        from dipy.align.reslice import reslice
        
        voxel_size = mri_resolution
        voxel_size_defined = True
        mri_resolution = True

    # Handle source space and resampling
    if isinstance(src, SourceSpaces):
        offset = 2 if src.kind == "mixed" else 0
        if voxel_size_defined:
            raise ValueError("Cannot infer original voxel size for reslicing. Set mri_resolution to a boolean value or apply morph first.")
            
        # Now deal with the fact that we may have multiple sub-volumes
        inuse = [s["inuse"] for s in src[offset:]]
        src_shape = [s["shape"] for s in src[offset:]]
        assert len(set(map(tuple, src_shape))) == 1
        src_subject = src._subject
        src = BunchConst(src_data=_get_src_data(src, mri_resolution)[0])
    else:
        # Make a list as we may have many inuse when using multiple sub-volumes
        inuse = src.src_data["inuse"]
        src_subject = src.subject_from
    assert isinstance(inuse, list)
    if stc.subject is not None:
        _check_subject_src(stc.subject, src_subject, "stc.subject")

    shape = src.src_data["src_shape"][::-1]  # Flip shape for NIfTI
    dtype = np.complex128 if np.iscomplexobj(stc.data) else np.float64
    vols = np.zeros((np.prod(shape[:3])), dtype=dtype, order="F")
    n_vertices_seen = 0
    for this_inuse in inuse:
        this_inuse = this_inuse.astype(bool)
        n_vertices = np.sum(this_inuse)
        stc_slice = slice(n_vertices_seen, n_vertices_seen + n_vertices)
        vols[this_inuse] = stc.data[stc_slice, tp]
        n_vertices_seen += n_vertices
    
    # use mri resolution as represented in src
    if mri_resolution:
        if src.src_data["interpolator"] is None:
            raise RuntimeError("Cannot morph with mri_resolution when add_interpolator=False was used with setup_volume_source_space")
        shape = src.src_data["src_shape_full"][::-1]
        vols = src.src_data["interpolator"] @ vols

    # reshape back to proper shape
    vols = np.reshape(vols, shape, order="F")
    
    # set correct space
    if mri_resolution:
        affine = src.src_data["src_affine_vox"]
    else:
        affine = src.src_data["src_affine_src"]
    
    affine = np.dot(src.src_data["src_affine_ras"], affine)
    affine[:3] *= 1e3 # Convert to mm
    
    # pre-define header
    header = NiftiHeader()
    header.set_xyzt_units("mm")
    header["pixdim"][4] = 1e3 * stc.tstep
    
    # if a specific voxel size was targeted (only possible after morphing)
    if voxel_size_defined:
        # reslice mri
        vols, affine = reslice(vols, affine, _get_zooms_orig(morph), voxel_size)
    
    with warnings.catch_warnings():  # nibabel<->numpy warning
        vols = NiftiImage(vols, affine, header=header)
        if fname is not None:
            nib.save(vols, fname)
    
    return vols


