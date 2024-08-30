"""
roi_utils.py

This module provides utility functions for handling label indices, creating region of interest (ROI) masks, and transforming atlases from template space to native subject space. These functions facilitate operations such as retrieving label indices, generating binary ROIs, and applying spatial transformations.

Functions:
- create_subject_rois(subject, atlas, local_dir, path_tflow_data, logger, plotting=False)
    Generate Regions of Interest (ROIs) for a given subject using a specified brain atlas and convert them from template 
    space to the subject's native space. This function handles various atlases, resamples the atlas to match the subject's 
    T1-weighted image, and transforms the atlas into the subject's native space.
    
- get_index_by_name(name, labels_df)
    Retrieve the index of a given label name from the labels DataFrame.

- create_roi(atlas_data, labels)
    Create a binary ROI from atlas data based on specified labels.

- transform_atlas_to_native_space(atlas_temp_space_path, T1_subject_path, T1_normalized_path, atlas_sub_space_path, local_dir, sub)
    Transform an atlas from MNI space to native subject space.

Variable:
ATLAS_REGION: Dict: A dictionary that specifies, for each atlas, the regions of interest (ROIs) to be created for each task of interest.
"""

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import ants
import nibabel as nib
from nilearn.image import resample_img, new_img_like
from nilearn.plotting import plot_roi
import matplotlib.pyplot as plt


ATLAS_REGIONS = {
    'AAL': {
        'aud': ['Heschl_L', 'Temporal_Sup_L', 'Heschl_R', 'Temporal_Sup_R'],
        'leftfoot': ['Paracentral_Lobule_L', 'Supp_Motor_Area_L'],
        'rightfoot': ['Paracentral_Lobule_R', 'Supp_Motor_Area_R'],
        'fing': ['Precentral_L', 'Supp_Motor_Area_L', 'Precentral_R', 'Supp_Motor_Area_R'],
        'vis': ['Calcarine_L', 'Cuneus_L', 'Lingual_L', 'Occipital_Sup_L', 'Occipital_Mid_L', 'Occipital_Inf_L', 'Fusiform_L', 'Calcarine_R', 'Cuneus_R', 'Lingual_R', 'Occipital_Sup_R', 'Occipital_Mid_R', 'Occipital_Inf_R', 'Fusiform_R']
    },
    
    'HOCPAL_dseg': {
        'aud': ['Left Planum Temporale', 'Right Planum Temporale'],
        'leftfoot': ['Left Precentral Gyrus', 'Left Postcentral Gyrus','Left Supplementary Motor Cortex'], 
        'rightfoot': ['Right Precentral Gyrus', 'Right Postcentral Gyrus', 'Right Supplementary Motor Cortex'],
        'fing': ['Left Precentral Gyrus', 'Left Postcentral Gyrus', 'Left Supplementary Motor Cortex', 'Right Precentral Gyrus', 'Right Postcentral Gyrus',  'Right Supplementary Motor Cortex'],
        'vis': ['Left Cuneal Cortex', 'Left Occipital Pole', 'Left Temporal Occipital Fusiform Cortex',
                  'Left Lateral Occipital Cortex Sup', 'Left Lateral Occipital Cortex Inf', 'Left Occipital Fusiform Gyrus',
               'Right Cuneal Cortex', 'Right Occipital Pole', 'Right Temporal Occipital Fusiform Cortex',
                  'Right Lateral Occipital Cortex Sup', 'Right Lateral Occipital Cortex Inf', 'Right Occipital Fusiform Gyrus']
    },
    
    'Schaefer2018_desc-100Parcels17Networks_dseg': {
        'aud': [
            '17Networks_LH_SomMotB_Aud_1', '17Networks_LH_SomMotB_Aud_2',
            '17Networks_LH_SomMotB_Aud_3', '17Networks_LH_SomMotB_Aud_4',
            '17Networks_RH_SomMotB_S2_1', '17Networks_RH_SomMotB_S2_2',
            '17Networks_RH_SomMotB_S2_3', '17Networks_RH_SomMotB_S2_4'
        ],
        'leftfoot': [
            '17Networks_LH_SomMotA_1', '17Networks_LH_SomMotA_2', '17Networks_LH_SomMotB_Aud_4'
        ],
        'rightfoot': [
            '17Networks_RH_SomMotA_1', '17Networks_RH_SomMotA_2',
            '17Networks_RH_SomMotA_3', '17Networks_RH_SomMotA_4', '17Networks_RH_SomMotB_S2_4'
        ],
        'fing': [
            '17Networks_LH_SomMotA_1', '17Networks_LH_SomMotA_2', '17Networks_LH_SomMotB_Aud_4',
            '17Networks_RH_SomMotA_1', '17Networks_RH_SomMotA_2',
            '17Networks_RH_SomMotA_3', '17Networks_RH_SomMotA_4', '17Networks_RH_SomMotB_S2_4'
        ],
        'vis': [
            '17Networks_RH_VisCent_ExStr_1', '17Networks_RH_VisCent_ExStr_2',
            '17Networks_RH_VisCent_ExStr_3', '17Networks_RH_VisCent_ExStr_4', '17Networks_LH_VisPeri_ExStrInf_1',
            '17Networks_LH_VisPeri_ExStrInf_2', '17Networks_LH_VisPeri_ExStrInf_3',
            '17Networks_RH_VisCent_ExStr_1', '17Networks_RH_VisCent_ExStr_2',
            '17Networks_RH_VisCent_ExStr_3', '17Networks_RH_VisPeri_ExStrSup_1',
            '17Networks_RH_VisPeri_ExStrSup_2', '17Networks_RH_VisPeri_ExStrSup_3'
        ]
    },

   'Schaefer2018_desc-100Parcels7Networks_dseg' : {
    'aud': [
        '7Networks_LH_SomMot_1', '7Networks_LH_SomMot_2',
        '7Networks_RH_SomMot_1', '7Networks_RH_SomMot_2'
    ],
    'leftfoot': [
        '7Networks_LH_SomMot_6', '7Networks_LH_SomMot_5', '7Networks_LH_SomMot_3',
        '7Networks_LH_SomMot_4'
    ],
    'rightfoot': [
        '7Networks_RH_SomMot_6', '7Networks_RH_SomMot_5', '7Networks_RH_SomMot_3',
        '7Networks_RH_SomMot_4', '7Networks_RH_SomMot_7', '7Networks_RH_SomMot_8'
    ],
    'fing': [
        '7Networks_LH_SomMot_6', '7Networks_LH_SomMot_5', '7Networks_LH_SomMot_3',
        '7Networks_LH_SomMot_4',
         '7Networks_RH_SomMot_6', '7Networks_RH_SomMot_5', '7Networks_RH_SomMot_3',
        '7Networks_RH_SomMot_4', '7Networks_RH_SomMot_7', '7Networks_RH_SomMot_8'
    ],

    'vis': [
        '7Networks_LH_Vis_1', '7Networks_LH_Vis_2', '7Networks_LH_Vis_3',
        '7Networks_LH_Vis_4', '7Networks_LH_Vis_5', '7Networks_LH_Vis_6',
        '7Networks_LH_Vis_7', '7Networks_LH_Vis_8',
        '7Networks_RH_Vis_1', '7Networks_RH_Vis_2', '7Networks_RH_Vis_3',
        '7Networks_RH_Vis_4', '7Networks_RH_Vis_5', '7Networks_RH_Vis_6',
        '7Networks_RH_Vis_7', '7Networks_RH_Vis_8'
    ]
}
}

def create_subject_rois(subject, atlas, local_dir, path_tflow_data, logger, plotting=False):
    """
    Generate Regions of Interest (ROIs) for a given subject using a specified brain atlas and convert them from template 
    space to the subject's native space. This function handles various atlases, resamples the atlas to match the subject's 
    T1-weighted image, and transforms the atlas into the subject's native space.

    Parameters:
    -----------
    subject : str
        The subject identifier for which the ROIs are being created.
    
    atlas : str
        The atlas name to be used for ROI creation. Available options include:
        'HOCPAL_desc-th0_dseg', 'HOCPAL_desc-th25_dseg', 'Schaefer2018_desc-100Parcels7Networks_dseg', 
        'Schaefer2018_desc-100Parcels17Networks_dseg', and 'AAL'.
    
    local_dir : str
        Path to the local directory where the subject's data is stored and where the output will be saved.
    
    path_tflow_data : str
        Path to the template flow data directory, containing the atlas and label files.
    
    logger : logging.Logger
        Logger instance for logging the process and capturing any errors that occur during execution.
    
    plotting : bool, optional (default=False)
        If True, the function will generate and display plots of the ROIs and atlas images in the subject's space.
    
    Raises:
    -------
    ValueError
        If the specified atlas is not available in the predefined list of atlases.
    
    Exception
        Logs and raises any exceptions that occur during the ROI creation process.
    
    """
    try:
        # List of available atlases
        atlas_available = ['HOCPALth0', 'HOCPALth25', 'Schaefer7',
                   'Schaefer17', 'AAL']

        atlas_mapping = {
            'HOCPALth0': 'HOCPAL_desc-th0_dseg',
            'HOCPALth25': 'HOCPAL_desc-th25_dseg',
            'Schaefer7': 'Schaefer2018_desc-100Parcels7Networks_dseg',
            'Schaefer17': 'Schaefer2018_desc-100Parcels17Networks_dseg',
            'AAL': 'AAL'
        }
        # Get the name of the atlas in template flow
        atlas_id = atlas_mapping.get(atlas, atlas)
        
        # Validate atlas input
        if atlas not in atlas_available:
            raise ValueError(f"Atlas should be one of {atlas_available}.")
    
        print(f"Creating ROIs in {subject} space from atlas {atlas}.")
        logger.info(f"Creating ROIs in {subject} space from atlas {atlas}.")

        # Define the derivatives directory for the running subject 
        sub_derivatives_dir = Path(local_dir) / 'derivatives' / subject
    
        # Define the path to the normalized T1-weighted anatomical data
        anat_outdir = sub_derivatives_dir / 'anat'
        T1_MNI_path = anat_outdir / f"{subject}_space-MNI152_T1w.nii.gz"
        T1_MNI_img = nib.load(T1_MNI_path)  # Load the T1 image
    
        # Define labels name based on atlas
        labels_name = 'HOCPAL_dseg' if 'HOCPAL' in atlas_id else atlas_id
        
        if atlas != 'AAL':
            # Load atlas and labels from predefined paths
            atlas_path = Path(path_tflow_data) / f'tpl-MNI152NLin6Asym_res-01_atlas-{atlas_id}.nii.gz'
            path_atlas_labels = Path(path_tflow_data) / f'tpl-MNI152NLin6Asym_atlas-{labels_name}.tsv'
            labels_df = pd.read_csv(path_atlas_labels, sep='\t')  # Load the labels
        else:
            # Fetch AAL atlas if it's not downloaded
            from nilearn import datasets
            dataset_aal = datasets.fetch_atlas_aal()
            atlas_path = dataset_aal.maps
    
        # Load the atlas image
        atlas_img = nib.load(atlas_path)
    
        # Resample the atlas to match T1 image dimensions and affine transformation
        atlas_img_resampled = resample_img(
            atlas_img, target_affine=T1_MNI_img.affine, 
            target_shape=T1_MNI_img.shape, interpolation='nearest' if atlas == 'AAL' else 'continuous'
        )
        
        # Define the output directory and file path for the resampled atlas
        atlas_sub_outdir = sub_derivatives_dir / 'anat' / f'atlas-{atlas}'
        atlas_sub_outdir.mkdir(parents=True, exist_ok=True)
        if atlas != 'AAL':
            atlas_sub_temp_space_path = atlas_sub_outdir / f"{subject}_space-MNI152NLin6Asym_atlas-{atlas}.nii.gz"
        else:
            atlas_sub_temp_space_path = atlas_sub_outdir / f"{subject}_space-MNI152_atlas-{atlas}.nii.gz"
        
        # Save the resampled atlas image if it doesn't already exist
        if not atlas_sub_temp_space_path.exists():
            nib.save(atlas_img_resampled, atlas_sub_temp_space_path)

        # Handle missing labels in HOCPAL atlas
        if 'HOCPAL' in atlas:
            missing_labels = {
                42: 'Left Lateral Occipital Cortex Sup',
                43: 'Right Lateral Occipital Cortex Sup',
                44: 'Left Lateral Occipital Cortex Inf',
                45: 'Right Lateral Occipital Cortex Inf',
                50: 'Left Supplementary Motor Cortex',
                51: 'Right Supplementary Motor Cortex'
            }
            
            # Add missing labels if they are not in the atlas data
            for label_index, label_name in missing_labels.items():
                if not any(labels_df['index'] == label_index):
                    missing_labels_df = pd.DataFrame(list(missing_labels.items()), columns=['index', 'name'])
                    labels_df = pd.concat([labels_df, missing_labels_df]).drop_duplicates(subset='index').sort_values(by='index').reset_index(drop=True)
                    labels_df.to_csv(path_atlas_labels, sep='\t', index=False)
    
        # Retrieve atlas indices based on the atlas type
        Tasks_ParcelLabels = ATLAS_REGIONS[labels_name]
        if atlas != 'AAL':
            Tasks_ParcelIndices = {task: [float(get_index_by_name(roi_, labels_df)) for roi_ in rois] for task, rois in Tasks_ParcelLabels.items()}
        elif 'HOCPAL' in atlas:
            Tasks_ParcelIndices = {task: [float(get_index_by_name(roi_, labels_df)) + 1 for roi_ in rois] for task, rois in Tasks_ParcelLabels.items()}
        else:
            Tasks_ParcelIndices = {task: [int(dataset_aal.indices[dataset_aal.labels.index(roi_)]) for roi_ in rois] for task, rois in Tasks_ParcelLabels.items()}
    
        # Define the path to the subject's T1 image in native space
        anat_source_path = Path(local_dir) / 'bids' / subject / 'anat'
        T1_sub_space_path = anat_source_path / f"{subject}_T1w.nii.gz"
        
        # Define the path for the atlas in subject space
        atlas_sub_space_path = atlas_sub_outdir / f"{subject}_atlas-{atlas}.nii.gz"
        
        # Transform the ROI from template to native space
        transform_atlas_to_native_space(atlas_sub_temp_space_path, T1_sub_space_path, T1_MNI_path, atlas_sub_space_path, local_dir, subject)
    
        # Optionally plot the resampled atlas in subject space
        if plotting:
            plot_roi(nib.load(atlas_sub_space_path), bg_img=nib.load(T1_sub_space_path), title=f'{atlas} in subject space', display_mode='ortho', draw_cross=True)
            plt.show()
            plt.close()
    
        # Define the output directory for the ROIs
        roi_sub_outdir = atlas_sub_outdir / 'rois'
        roi_sub_outdir.mkdir(parents=True, exist_ok=True)
        
        # Load the resampled atlas in subject space
        atlas_sub_space = nib.load(atlas_sub_space_path)
        
        # Create and save each ROI for each task
        for task, indices in Tasks_ParcelIndices.items():
            roi_path = roi_sub_outdir / f"{subject}_atlas-{atlas}_roi-{task}.nii.gz"
            
            if not roi_path.exists():
                # Generate the ROI data
                roi_data = create_roi(atlas_sub_space.get_fdata(), indices)
                roi_img = new_img_like(atlas_sub_space, roi_data)
                nib.save(roi_img, roi_path)
            else:
                roi_img = nib.load(roi_path)
            
            # Optionally plot the ROI
            if plotting:
                plot_roi(roi_img, bg_img=nib.load(T1_sub_space_path), title=f'{task} ROI', display_mode='ortho', draw_cross=True)
                plt.show()
                plt.close()

    except Exception as e:
        raise e

def get_index_by_name(name, labels_df):
    """Retrieve the index of a given label name from the labels DataFrame.
    
    Args:
        name (str): The label name to look up.
        labels_df (pd.DataFrame): DataFrame containing the label names and indices.
    
    Returns:
        int: The index of the label
    """
    result = labels_df[labels_df['name'] == name]
    if not result.empty:
        return result['index'].values[0]
    else:
        raise ValueError(f'No index found for parcel  {name}.') 

def create_roi(atlas_data, labels):
    """
    Create a binary roi from an atlas data based on specified labels.

    Parameters:
    ----------
    atlas_data : numpy.ndarray
        The atlas data where each voxel value corresponds to a specific anatomical region.
    labels : list of int
        A list of labels indicating the regions of interest to be included in the mask.

    Returns:
    -------
    roi_data : numpy.ndarray
        A binary roi with the same shape as atlas_data, where voxels belonging to the specified labels are set to 1 and all other voxels are set to 0.
    """
    roi_data = np.zeros(atlas_data.shape)
    for label in labels:
        roi_data[atlas_data == label] = 1
    return roi_data

def transform_atlas_to_native_space(atlas_temp_space_path: Path, T1_subject_path: Path, T1_normalized_path: Path, atlas_sub_space_path: Path, local_dir, sub: str):
    """
    Transform an atlas from MNI space to native subject space.

    Parameters:
    atlas_temp_space_path (Path): Path to the atlas in a template space (same as T1_normalized_path).
    T1_subject_path (Path): Path to the native T1-weighted anatomical image.
    T1_normalized_path (Path): Path to the normalized T1-weighted anatomical image (same space as atlas_temp_space_path).
    atlas_sub_space_path (Path): Path to save the atlas in the subject space (same as T1_subject_path).
    local_dir (Path or str): Local directory for saving outputs.
    sub (str): Subject identifier.

    Return:
    atlas_sub_space_path (Path): Path to the atlas in subject space

    Raises:
    ValueError: If any of the paths are not instances of Path.
    FileNotFoundError: If any of the specified files do not exist.
    """
    
    # Type checking
    if not isinstance(atlas_temp_space_path, Path) or not isinstance(T1_subject_path, Path) or not isinstance(T1_normalized_path, Path) or not isinstance(atlas_sub_space_path, Path):
        raise TypeError("atlas_temp_space_path, T1_subject_path, T1_normalized_path, atlas_sub_space_path must be of type Path.")
    
    # Define the output directory for transformation information
    anat_outdir =  Path(local_dir) / 'derivatives' / sub / 'anat' / 'transform_xfm'
    anat_outdir.mkdir(parents=True, exist_ok=True)
    
    # Load the T1 images with ANTs
    T1_subject_img = ants.image_read(str(T1_subject_path))
    T1_normalized_img = ants.image_read(str(T1_normalized_path))

    # Perform registration
    # To not run the transformation each times
    if not (anat_outdir / f'{sub}_ants_template2subspace_Composite.h5').exists() or not (anat_outdir / f'{sub}_ants_template2subspace_InverseComposite.h5').exists():
        # Perform registration
        xfm = ants.registration(fixed=T1_subject_img, moving=T1_normalized_img, type_of_transform='SyN', write_composite_transform=True, outprefix=str(anat_outdir / f'{sub}_ants_template2subspace_'))
    
    else:
        xfm = {}
        xfm['fwdtransforms'] = str(anat_outdir / f'{sub}_ants_template2subspace_Composite.h5')
    
    # Load the ROI with ANTs
    roi_img = ants.image_read(str(atlas_temp_space_path))
    
    # Transform the ROI to native space
    ROI_in_native_space = ants.apply_transforms(fixed=T1_subject_img, moving=roi_img, transformlist=xfm['fwdtransforms'], interpolator='nearestNeighbor')
    
    # Save the transformed ROI
    ROI_in_native_space.to_filename(atlas_sub_space_path)

    print(f"Transformed atlas saved to: {atlas_sub_space_path}")

    return(atlas_sub_space_path)