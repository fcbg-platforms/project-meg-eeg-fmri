import sys
import nibabel as nib
from pathlib import Path
import re
import pandas as pd
import json
import os

# Add the directory that contains the utils package to sys.path
sys.path.append(str(Path('..').resolve()))

from config import PATH_TO_MRI_DATA, LOCAL_DIR, MRI_DATA_CONV_INFO_DIR, SUBJECTS
from utils.utils import setup_logging

# Define which parts of the script to run
RUN_T1TEMP2BIDS = True
RUN_FMRICONTRAST2BIDS = True
RUN_SPMCON2BIDS = True
RUN_SPMBETA2BIDS = True
RUN_RESMS2BIDS = True
RUN_MASK2BIDS = True
RUN_RPV2BIDS = True

# Initialize the logger
logger = setup_logging(LOCAL_DIR, 'logger-mri2bids')

# Mapping between task names in MRI_analyses/Results and their BIDS equivalents
CORRESP_TASK_BIDS = {
    "aud": 'Auditory',
    "aud_all": 'AuditoryAll',
    "feet": 'Feet',
    "fing": 'Hand',
    "vis": 'Visual'
}

# Mapping between task names in MRI_analyses/Nifti and their BIDS equivalents for BOLD data
CORRESP_TASK_BIDS_BOLD = {
    "aud": 'sparseAuditory',
    "feet": 'feet',
    "fing": 'hand',
    "vis": 'face'
}

# Define contrasts associated with each task in BIDS format
TASKS_CONTRASTS = {
    "aud": ['sine', 'piano1', 'piano4', 'div', 'piano1_min_sine', 'piano4_min_sine', 'piano4_min_piano1'],
    "aud_all": ['audioStim'],
    "feet": ['leftfoot', 'rightfoot', 'L_min_R', 'R_min_L'],
    "fing": ['finger'],
    "vis": ['face', 'scramble', 'face_min_scramble', 'house']
}

# Define regressors associated with each task in BIDS format
TASKS_REGRESSORS = {
    "aud": ['SineHrf', 'SinedHrf', 'Piano1Hrf', 'Piano1dHrf', 'Piano4Hrf', 'Piano4dHrf', 'DivHrf', 'DivdHrf', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'Constant'],
    "aud_all": ['AudioAllHrf', 'AudioAlldHrf', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'Constant'],
    "feet": ['LeftHrf', 'LeftdHrf', 'RightHrf', 'RightdHrf', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'Constant'],
    "fing": ['HandHrf', 'HanddHrf' , 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'Constant'],
    "vis": ['FacesHrf', 'FacesdHrf', 'ScrambleHrf',  'ScrambledHrf', 'HouseHrf', 'HousedHrf', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'Constant']
}

# Input validation
if not SUBJECTS or not isinstance(SUBJECTS, list):
    logger.error("Invalid 'SUBJECTS' input. It should be a non-empty list.")
    raise ValueError("Invalid 'SUBJECTS' input. It should be a non-empty list.")

if not Path(LOCAL_DIR).exists():
    logger.error(f"Directory '{LOCAL_DIR}' does not exist.")
    raise FileNotFoundError(f"Directory '{LOCAL_DIR}' does not exist.")

if not Path(MRI_DATA_CONV_INFO_DIR).exists():
    logger.error(f"Directory '{MRI_DATA_CONV_INFO_DIR}' does not exist.")
    raise FileNotFoundError(f"Directory '{MRI_DATA_CONV_INFO_DIR}' does not exist.")
    

## Transformation of T1w in SPM template space in BIDS format
if RUN_T1TEMP2BIDS == True:
    """
    Save the normalized T1-weighted anatomical data in BIDS format for a list of subjects.
    """
    print("\nStarting conversion of T1-weighted template to BIDS format.")
    logger.info("Starting conversion of T1-weighted template to BIDS format.")
    
    for sub in SUBJECTS:
        # Define the directory path for normalized T1-weighted anatomical data
        directory_path = Path(PATH_TO_MRI_DATA) / f'Subj-0{sub[-2:]}' / 'NIfTI'
        
        # Define the output directory for the BIDS-formatted anatomical data
        anat_outdir = Path(LOCAL_DIR)/ 'derivatives' / sub /'anat'
        anat_outdir.mkdir(parents=True, exist_ok=True)
        
        # Define the expected output path for the BIDS-formatted T1-weighted image
        T1_MNI_BIDS_path = anat_outdir / f"{sub}_space-MNI152_T1w.nii.gz"

        # If the BIDS-formatted image does not exist, proceed with the conversion
        if not T1_MNI_BIDS_path.exists():
            # Search for the normalized T1-weighted image produced by SPM
            file_pattern = 'w*_t1_mprage_sag_p2_iso_Fov256.nii'
            matching_files = list(directory_path.glob(file_pattern))
            
            # Validate the search results
            if len(matching_files) == 1:
                T1_MNI_path = matching_files[0]
            elif len(matching_files) == 0:
                logger.error(f"  No matching T1-weighted files found for subject {sub}.")
                raise FileNotFoundError(f"No matching T1-weighted files found for subject {sub}.")
            else:
                logger.error(f"  Multiple matching T1-weighted files found for subject {sub}.")
                raise ValueError(f"Multiple matching T1-weighted files found for subject {sub}.")
                
            # Load the image and save it in BIDS format
            T1_MNI_img = nib.load(T1_MNI_path)
            nib.save(T1_MNI_img, T1_MNI_BIDS_path)
            logger.info(f"  Saved {T1_MNI_BIDS_path}")
            print(f"  Saved {T1_MNI_BIDS_path}")

        else:
            print(f"  {T1_MNI_BIDS_path} already exists, skipping.")

    print(f'Conversion of T1-weighted template to BIDS format completed successfully for subjects: {SUBJECTS}.')
    logger.info(f'Conversion of T1-weighted template to BIDS format completed successfully for subjects: {SUBJECTS}.')
                
## Transformation of fMRI Data SPM contrasts to BIDS Formalism
if RUN_FMRICONTRAST2BIDS == True:
    """
    Converts fMRI contrast data to BIDS (Brain Imaging Data Structure) formalism for a list of subjects.

    This function processes MRI contrast data for specified subjects, converts it into BIDS-compliant 
    format, and saves the converted data in the appropriate directories. The conversion is based on 
    predefined MRI contrast tasks and their corresponding BIDS names, as well as a correspondence file 
    that maps contrasts to specific SPM names.
    """
    print('\nConversion of fMRI data SPM contrasts to BIDS format :')
    logger.info('Conversion of fMRI data SPM contrasts to BIDS format :')
    
    # Iterate through subjects and tasks
    for sub in SUBJECTS:
        for task, contrasts in TASKS_CONTRASTS.items():
            try:
                # Define the output directory and create it if it doesn't exist
                t_bids_outdir = Path(LOCAL_DIR) / 'derivatives' / sub / 'func' / f'task-{task}'
                t_bids_outdir.mkdir(parents=True, exist_ok=True)
                
                # Read the correspondence file for the current task
                file_path = Path(MRI_DATA_CONV_INFO_DIR) / f'Corresp_SPM_con_{task}.xlsx'
                
                if not file_path.exists():
                    logger.warning(f"    Correspondence file '{file_path}' not found. Skipping task '{task}'.")
                    raise FileNotFoundError(f"    Correspondence file '{file_path}' not found.")
                
                corresp_spmT_con = pd.read_excel(file_path)
                
                for contrast in contrasts:
                    # Define the output path in BIDS format
                    t_bids_path = t_bids_outdir / f"{sub}_task-{task}_contrast-{contrast}_desc-stat-t_statmap.nii.gz"

                    if not t_bids_path.exists(): 
                        # Get the info of this contrast
                        info_contrast = corresp_spmT_con.loc[corresp_spmT_con['Contrast'] == contrast]
                        
                        if info_contrast.empty:
                            logger.error(f"    No contrast '{contrast}' found in the correspondence file '{file_path}'.")
                            raise ValueError(f"    No contrast '{contrast}' found in the correspondence file '{file_path}'.")
                        
                        # Get the correspondence
                        con_corresp = info_contrast['SPM_name'].values[0]
                        
                        # Path to the fMRI contrast data
                        path_fmri_contrast = Path(PATH_TO_MRI_DATA) / f'Subj-0{sub[-2:]}' / 'Results-NativeSpace' / CORRESP_TASK_BIDS[task]/ f'{con_corresp}.nii'
                        
                        if not path_fmri_contrast.exists():
                            logger.error(f"    fMRI contrast file '{path_fmri_contrast}' not found. Skipping contrast '{contrast}'.")
                            raise FileNotFoundError(f"    fMRI contrast file '{path_fmri_contrast}' not found.")
                        
                        # Load the fMRI contrast data
                        fmri_contrast = nib.load(path_fmri_contrast)
                        
                        # Save the contrast data in the new location
                        nib.save(fmri_contrast, t_bids_path)
                        logger.info(f"    T-stats contrast '{con_corresp}' saved in BIDS formalism at {t_bids_path}")
                        print(f"    T-stats contrast '{con_corresp}' saved in BIDS formalism at {t_bids_path}")
            
            except Exception as e:
                logger.error(f"    Error processing subject '{sub}' for task '{task}': {e}")
                raise e
    
    logger.info(f"Conversion of fMRI data SPM contrasts to BIDS format completed for subjects: {SUBJECTS}.")
    print(f"Conversion of fMRI data SPM contrasts to BIDS format completed for subjects: {SUBJECTS}.")

# Convert SPM con output to BIDS format.
if RUN_SPMCON2BIDS == True:
    """
    Convert SPM contrast results (`con_xxxx.nii`) to BIDS format for a list of subjects.

    The `con_xxxx.nii` files from SPM represent contrast images that show the linear combinations of beta estimates for the defined contrasts between regressors.

    This function processes the contrast images for specified subjects, converts them to BIDS-compliant format, and saves them in the appropriate directories. The conversion relies on predefined MRI contrast tasks and their corresponding BIDS names, as well as a correspondence file that maps SPM contrast names to specific BIDS names.
    """
    
    print('\nConverting SPM `con_xxxx.nii` results to BIDS format:')
    logger.info('Converting SPM `con_xxxx.nii` results to BIDS format:')

    # Iterate through subjects and tasks
    for sub in SUBJECTS:
        for task, contrasts in TASKS_CONTRASTS.items():
            try:
                # Define and create the output directory if it doesn't exist
                con_bids_outdir = Path(LOCAL_DIR) / 'derivatives' / sub / 'func' / f'task-{task}'
                con_bids_outdir.mkdir(parents=True, exist_ok=True)
                
                # Read the correspondence file for the current task
                file_path = Path(MRI_DATA_CONV_INFO_DIR) / f'Corresp_SPM_con_{task}.xlsx'
                
                if not file_path.exists():
                    logger.warning(f"    Correspondence file '{file_path}' not found. Skipping task '{task}'.")
                    print(f"    Correspondence file '{file_path}' not found. Skipping task '{task}'.")
                    continue
                
                corresp_spmT_con = pd.read_excel(file_path)
                
                for contrast in contrasts:
                    # Define the output path in BIDS format
                    con_bids_path = con_bids_outdir / f"{sub}_task-{task}_contrast-{contrast}_desc-stat-con_statmap.nii.gz"

                    if not con_bids_path.exists(): 
                        # Get the info of this contrast
                        info_contrast = corresp_spmT_con[corresp_spmT_con['Contrast'] == contrast]
                        
                        if info_contrast.empty:
                            logger.error(f"    No contrast '{contrast}' found in the correspondence file '{file_path}'.")
                            print(f"    No contrast '{contrast}' found in the correspondence file '{file_path}'.")
                            continue
                        
                        # Get the corresponding SPM name
                        con_corresp = info_contrast['SPM_name'].values[0]
                        
                        # Path to the `con_xxxx` data
                        path_fmri_contrast =Path(PATH_TO_MRI_DATA) / f'Subj-0{sub[-2:]}' / 'Results-NativeSpace' / CORRESP_TASK_BIDS[task]/ f'con_{con_corresp[-4:]}.nii'
                        
                        if not path_fmri_contrast.exists():
                            logger.error(f"    Contrast file '{path_fmri_contrast}' not found.")
                            print(f"    Contrast file '{path_fmri_contrast}' not found.")
                            continue
                        
                        # Load and save the contrast data
                        fmri_contrast = nib.load(path_fmri_contrast)
                        nib.save(fmri_contrast, con_bids_path)
                        logger.info(f"    Contrast 'con_{con_corresp[-4:]}' saved in BIDS format at {con_bids_path}")
                        print(f"    Contrast 'con_{con_corresp[-4:]}' saved in BIDS format at {con_bids_path}")
            
            except Exception as e:
                logger.error(f"    Error processing subject '{sub}' for task '{task}': {e}")
                raise e

    print(f'Conversion of SPM `con_xxxx.nii` results to BIDS format completed successfully for subjects {SUBJECTS}.')
    logger.info(f'Conversion of SPM `con_xxxx.nii` results to BIDS format completed successfully for subjects {SUBJECTS}.')


# Convert SPM beta output to BIDS format.
if RUN_SPMBETA2BIDS == True:
    """
    Convert SPM beta estimates results (`beta_xxxx.nii`) to BIDS format for a list of subjects.

    The `beta_xxxx.nii` files from SPM represent beta estimates obtained during GLM analysis for the specified regressors.

    This function processes beta images for the specified subjects, converts them to BIDS-compliant format, and saves them in the appropriate directories. The conversion uses predefined regressors and tasks and their corresponding BIDS names, as well as a correspondence file that maps SPM beta names to specific BIDS names.
    """

    print('\nConverting SPM `beta_xxxx.nii` results to BIDS format:')
    logger.info('Converting SPM `beta_xxxx.nii` results to BIDS format:')

    # Iterate through subjects and tasks
    for sub in SUBJECTS:
        for task, regressors in TASKS_REGRESSORS.items():
            try:
                # Define and create the output directory if it doesn't exist
                beta_bids_outdir = Path(LOCAL_DIR) / 'derivatives' / sub / 'func' / f'task-{task}'
                beta_bids_outdir.mkdir(parents=True, exist_ok=True)
                
                # Read the correspondence file for the current task
                file_path = Path(MRI_DATA_CONV_INFO_DIR) / f'Corresp_beta_regressors_{task}.xlsx'

                if not file_path.exists():
                    logger.warning(f"    Correspondence file '{file_path}' not found. Skipping task '{task}'.")
                    print(f"    Correspondence file '{file_path}' not found. Skipping task '{task}'.")
                    continue
                
                corresp_beta_reg = pd.read_excel(file_path)

                for regressor in regressors:
                    # Define the output path in BIDS format
                    beta_bids_path = beta_bids_outdir / f"{sub}_task-{task}_desc-regressor-{regressor}_betamap.nii.gz"

                    if not beta_bids_path.exists(): 
                        # Get the information for this regressor
                        info_reg = corresp_beta_reg[corresp_beta_reg['Regressor'] == regressor]
                        
                        if info_reg.empty:
                            logger.error(f"    No regressor '{regressor}' found in the correspondence file '{file_path}'.")
                            print(f"    No regressor '{regressor}' found in the correspondence file '{file_path}'.")
                            continue
                        
                        # Get the corresponding SPM name
                        beta_corresp = info_reg['SPM_name'].values[0]
                        
                        # Path to the beta_xxxx data
                        path_beta =Path(PATH_TO_MRI_DATA) / f'Subj-0{sub[-2:]}' / 'Results-NativeSpace' / CORRESP_TASK_BIDS[task]/ f'{beta_corresp}.nii'
                        
                        if not path_beta.exists():
                            logger.error(f"    Beta file '{path_beta}' not found.")
                            print(f"    Beta file '{path_beta}' not found.")
                            continue
                        
                        # Load and save the beta data
                        beta_map = nib.load(path_beta)
                        nib.save(beta_map, beta_bids_path)
                        logger.info(f"    Beta map '{beta_corresp}' saved in BIDS format at {beta_bids_path}")
                        print(f"    Beta map '{beta_corresp}' saved in BIDS format at {beta_bids_path}")
            
            except Exception as e:
                logger.error(f"    Error processing subject '{sub}' for task '{task}': {e}")
                raise e

    print(f'Conversion of SPM `beta_xxxx.nii` results to BIDS format completed successfully for subjects {SUBJECTS}.')
    logger.info(f'Conversion of SPM `beta_xxxx.nii` results to BIDS format completed successfully for subjects {SUBJECTS}.')

# Convert SPM Residual Mean Square (ResMS) output to BIDS format.
if RUN_RESMS2BIDS == True:
    """
    Convert SPM residual mean square results (`ResMS.nii`) to BIDS format for a list of subjects.
    """
    
    print('\nConverting SPM `ResMS.nii` results to BIDS format:')
    logger.info('Converting SPM `ResMS.nii` results to BIDS format:')

    # Iterate through subjects and tasks
    for sub in SUBJECTS:
        for task, contrasts in TASKS_CONTRASTS.items():
            try:
                # Define and create the output directory if it doesn't exist
                resms_bids_outdir = Path(LOCAL_DIR) / 'derivatives' / sub / 'func' / f'task-{task}'
                resms_bids_outdir.mkdir(parents=True, exist_ok=True)

                # Define the output path in BIDS format
                resms_bids_path = resms_bids_outdir / f"{sub}_task-{task}_desc-stat-ResMS_statmap.nii.gz"

                if not resms_bids_path.exists(): 
                    # Path to the ResMS data
                    path_ResMS = Path(PATH_TO_MRI_DATA) / f'Subj-0{sub[-2:]}' / 'Results-NativeSpace' / CORRESP_TASK_BIDS[task]/ 'ResMS.nii'
                    
                    if not path_ResMS.exists():
                        logger.error(f"    Residual Mean Square file '{path_ResMS}' not found.")
                        print(f"    Residual Mean Square file '{path_ResMS}' not found.")
                        continue
                    
                    # Load and save the ResMS data
                    resMS = nib.load(path_ResMS)
                    nib.save(resMS, resms_bids_path)
                    logger.info(f"    Residual Mean Square file saved in BIDS format at {resms_bids_path}")
                    print(f"    Residual Mean Square file saved in BIDS format at {resms_bids_path}")
            
            except Exception as e:
                logger.error(f"    Error processing subject '{sub}' for task '{task}': {e}")
                raise e

    print(f'Conversion of SPM `ResMS.nii` results to BIDS format completed successfully for subjects {SUBJECTS}.')
    logger.info(f'Conversion of SPM `ResMS.nii` results to BIDS format completed successfully for subjects {SUBJECTS}.')

# Convert SPM Estimated RESELS per voxels (RPV) output to BIDS format.
if RUN_RPV2BIDS == True:
    """
    Convert the estimated RESELS per voxels (`RPV.nii`) to BIDS format for a list of subjects.
    """
    
    print('\nConverting SPM `RPV.nii` results to BIDS format:')
    logger.info('Converting SPM `RPV.nii` results to BIDS format:')

    # Iterate through subjects and tasks
    for sub in SUBJECTS:
        for task, contrasts in TASKS_CONTRASTS.items():
            try:
                # Define and create the output directory if it doesn't exist
                rpv_bids_outdir = Path(LOCAL_DIR) / 'derivatives' / sub / 'func' / f'task-{task}'
                rpv_bids_outdir.mkdir(parents=True, exist_ok=True)

                # Define the output path in BIDS format
                rpv_bids_path = rpv_bids_outdir / f"{sub}_task-{task}_desc-stat-RPV_statmap.nii.gz"

                if not rpv_bids_path.exists(): 
                    # Path to the ResMS data
                    path_RPV = Path(PATH_TO_MRI_DATA) / f'Subj-0{sub[-2:]}' / 'Results-NativeSpace' / CORRESP_TASK_BIDS[task]/ 'RPV.nii'
                    
                    if not path_RPV.exists():
                        logger.error(f"    Estimated RESELS per voxels file '{path_RPV}' not found.")
                        print(f"    Estimated RESELS per voxels file '{path_RPV}' not found.")
                        continue
                    
                    # Load and save the RPV data
                    rpv = nib.load(path_RPV)
                    nib.save(rpv, rpv_bids_path)
                    logger.info(f"    Estimated RESELS per voxels file saved in BIDS format at {rpv_bids_path}")
                    print(f"    Estimated RESELS per voxels file saved in BIDS format at {rpv_bids_path}")
                    
            except Exception as e:
                logger.error(f"    Error processing subject '{sub}' for task '{task}': {e}")
                raise e

    print(f'Conversion of SPM `RPV.nii` results to BIDS format completed successfully for subjects {SUBJECTS}.')
    logger.info(f'Conversion of SPM `RPV.nii` results to BIDS format completed successfully for subjects {SUBJECTS}.')

# Convert SPM Brain Mask output to BIDS format.
if RUN_MASK2BIDS == True:
    """
    Convert the Brain Mask (`mask.nii`) of each task to BIDS format for a list of subjects.
    """
    
    print('\nConverting SPM `mask.nii` results to BIDS format:')
    logger.info('Converting SPM `mask.nii` results to BIDS format:')

    # Iterate through subjects and tasks
    for sub in SUBJECTS:
        for task, contrasts in TASKS_CONTRASTS.items():
            try:
                # Define and create the output directory if it doesn't exist
                mask_bids_outdir = Path(LOCAL_DIR) / 'derivatives' / sub / 'func' / f'task-{task}'
                mask_bids_outdir.mkdir(parents=True, exist_ok=True)

                # Define the output path in BIDS format
                mask_bids_path = mask_bids_outdir / f"{sub}_task-{task}_desc-brainmask.nii.gz"

                if not mask_bids_path.exists(): 
                    # Path to the ResMS data
                    path_mask = Path(PATH_TO_MRI_DATA) / f'Subj-0{sub[-2:]}' / 'Results-NativeSpace' / CORRESP_TASK_BIDS[task]/ 'mask.nii'
                    
                    if not path_mask.exists():
                        logger.error(f"    Brain Mask file '{path_mask}' not found.")
                        print(f"    Brain Mask file '{path_mask}' not found.")
                        continue
                    
                    # Load and save the mask data
                    mask = nib.load(path_mask)
                    nib.save(mask, mask_bids_path)
                    logger.info(f"    Brain Mask file saved in BIDS format at {mask_bids_path}")
                    print(f"    Brain Mask file saved in BIDS format at {mask_bids_path}")
            
            except Exception as e:
                logger.error(f"    Error processing subject '{sub}' for task '{task}': {e}")
                raise e

    print(f'Conversion of SPM `mask.nii` results to BIDS format completed successfully for subjects {SUBJECTS}.')
    logger.info(f'Conversion of SPM `mask.nii` results to BIDS format completed successfully for subjects {SUBJECTS}.')
    