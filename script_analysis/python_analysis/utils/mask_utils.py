"""mask_utils.py

This module contains function for processing neuroimaging data, specifically handling masks. It includes a method to extract brain mask from FreeSurfer ribbon files and a function to mask nifti by an roi.

Functions
-----------
process_ribbon(subject, path_ribbon, anat_outdir)
    Processes a FreeSurfer ribbon file for a given subject, creating a brain mask and saving the results as NIfTI files.
masking_data_w_roi(data_path, sub_derivatives_outdir, sub, task, condition, atlas):
    Apply a ROI mask to the given data and save the masked output.

"""
import nibabel as nib
from pathlib import Path
import numpy as np
from nilearn.maskers import NiftiMasker

def process_ribbon(subject, path_ribbon, anat_outdir):
    """
    Processes the ribbon file and creates a brain mask for a given subject. Saves the ribbon and brain mask as NIfTI files,
    and plots the brain mask.

    Parameters:
    subject (str): The subject identifier.
    path_ribbon (str): The directory containing the data for all subjects.
    anat_outdir (str): The output directory where the processed NIfTI files will be saved.

    Return:
    brain_mask_img (Nifti1Image): The brain mask
    """
    # Define the path to the ribbon file
    path_ribbon = Path(path_ribbon) / 'derivatives' / subject / 'freesurfer' / 'mri' / 'ribbon.mgz'

    # Load the ribbon file
    ribbon = nib.load(path_ribbon)

    # Save the ribbon as a NIfTI file
    ribbon_nifti_path = Path(anat_outdir) / f"{subject}_ribbon.nii.gz"
    if not ribbon_nifti_path.exists():
        nib.save(ribbon, ribbon_nifti_path)

    # Extract brain mask from the ribbon
    ribbon_data = ribbon.get_fdata()
    brain_mask_data = np.zeros(ribbon_data.shape)
    brain_mask_data[np.where(ribbon_data != 0)] = 1

    # Save the brain mask as a NIfTI file
    brain_mask_img = nib.Nifti1Image(brain_mask_data, ribbon.affine)
    path_brain_mask = Path(anat_outdir) / f"{subject}_brainmask.nii.gz"
    nib.save(brain_mask_img, path_brain_mask)

    return brain_mask_img
    

def masking_data_w_roi(data_path, sub_derivatives_outdir, sub, task, condition, atlas):
    """
    Apply a ROI mask to the given data and save the masked output.

    Parameters:
    - data_path (str or Path): Path to the data file to be masked.
    - sub_derivatives_outdir (str or Path): Directory where subject derivatives are stored.
    - sub (str): Subject identifier.
    - task (str): Task identifier.
    - condition (str): Condition identifier
    - atlas (str): Atlas identifier used to locate the ROI.

    Returns:
    - Path: Path to the saved masked data.
    """
    # Convert paths to Path objects if they are strings
    data_path = Path(data_path)
    sub_derivatives_outdir = Path(sub_derivatives_outdir)

    # Load the data
    try:
        data = nib.load(data_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading data from {data_path}: {e}")

    # Path to the atlas ROI for this subject
    rois_outdir = sub_derivatives_outdir / 'anat' / f'atlas-{atlas}' / 'rois'
    if task == 'feet':
        roi_path = rois_outdir / f'{sub}_atlas-{atlas}_roi-{condition}.nii.gz'
    else:
        roi_path = rois_outdir / f'{sub}_atlas-{atlas}_roi-{task}.nii.gz'

    # Check if ROI file exists
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    # Mask the data
    try:
        roi_nifti_masker = NiftiMasker(mask_img=nib.load(roi_path), target_affine = data.affine, target_shape = data.shape[:3])
        masked_data = roi_nifti_masker.inverse_transform(
            roi_nifti_masker.fit_transform(data)
        )

    except Exception as e:
        raise RuntimeError(f"Error during masking process: {e}")

    # Define and create the output directory
    if task == 'feet':
        masked_path = data_path.parent / (data_path.with_suffix('').stem + f'_atlas-{atlas}_roi-{condition}.nii.gz')
    else:
        masked_path = data_path.parent / (data_path.with_suffix('').stem + f'_atlas-{atlas}_roi-{task}.nii.gz')

    # Copy the description
    masked_data.header['descrip'] = data.header['descrip']
    # Save the masked data
    try:
        nib.save(masked_data, masked_path)
    except Exception as e:
        raise IOError(f"Error saving masked data to {masked_path}: {e}")

    return masked_path