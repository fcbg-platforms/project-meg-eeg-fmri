import logging
import sys
import time

# Import constants and configurations
from config import (
    LOCAL_DIR, PATH_TO_MRI_DATA, MRI_DATA_CONV_INFO_DIR, PATH_TO_TPLFLOW, R_WORKING_DIRECTORY, RSCRIPT_EXECUTABLE, SUBJECTS, 
    THRESHOLD_PARAMS, TASKS_CONDITIONS, EEG_TASKS_TP, EEG_STC_ALGO, MEG_TASKS_TP, MEG_STC_ALGO, DO_BRAIN_MASKING, POS,
    PLOTTING, MRI_RESOLUTION, ATLAS, FMRI_COMPARISON_CLUSTERS, ROI_MASKING, NB_PERMUTATIONS, DISPERSION_DIST_METRIC, 
    MEAN_COMPA_VALUE, MEAN_TYPE_ANALYSIS, DISPERSION_SCRIPT, CorrectionParam, 
    print_configuration
)

from utils.utils import *
from utils.stats_utils import *

# Define a log file
logger = setup_logging(LOCAL_DIR, 'logger-MEEG_fMRI_whole_comparison')

def choose_sections_to_run(sections):
    """
    Allow the user to select which sections to run from the list of available sections.
    """
    while True:
        choices = input("Enter the numbers of the sections you want to run, separated by commas: ")

        # Check if input is valid
        if not choices or not all(part.strip().isdigit() for part in choices.split(',')):
            print("Invalid input. Please enter numbers separated by commas.")
            
        else:
            try:
                # Convert the input into a list of integers, adjusting for zero-based index
                chosen_indices = [int(choice.strip()) - 1 for choice in choices.split(',')]
    
                # Validate all choices
                if all(0 <= index < len(sections) for index in chosen_indices):
                    # Run the chosen sections
                    for index in chosen_indices:
                        section_function, section_name = sections[index]
                        section_function()
                        
                    break # Exit loop if valid input
                
                else:
                    # Handle invalid index
                    invalid_indices = [index + 1 for index in chosen_indices if not (0 <= index < len(sections))]
                    print(f"Invalid choices: {', '.join(map(str, invalid_indices))}. Please choose numbers from 1 to {len(sections)}.")
            
            except Exception as e:
                # Handle invalid input
                raise e    
    

def fmri_data_preparation():
    """
    Initializes the fMRI data thresholding process.
    
    This function:
    1. Starts by logging the beginning of the fMRI data preparation process.
    2. Sets up a logger to capture detailed logs specific to fMRI data preparation.
    3. Calls the mri_data_thresholding function to convert tstats in zscores, and threshold them.
    4. Handles any exceptions that occur during the process, logging the error details.
    """
    from utils.mri_utils import mri_data_thresholding
    
    # Log the start of the fMRI data preparation
    print("\nStarting fMRI data preparation...")
    logger.info("Starting fMRI data preparation...")
    logger.info("  Detailed logs for this analysis will be saved to 'logger-fmri_data_thresholding.log'.")
    
    # Set up logging for EEG data preparation
    logger_fmri = setup_logging(LOCAL_DIR, 'logger-fmri_data_thresholding')
    try:
        # Process fMRI data
        mri_data_thresholding(LOCAL_DIR, SUBJECTS, TASKS_CONDITIONS, logger_fmri, plotting = False)
    except Exception as e:
        # Log any errors that occur during the data preparation
        logger.error(f"  An error occurred during `mri_data_thresholding`: {e}")
        raise

def eeg_data_preparation():
    """
    Initializes the EEG data preparation process.
    
    This function:
    1. Starts by logging the beginning of the EEG data preparation process.
    2. Sets up a logger to capture detailed logs specific to EEG data preparation.
    3. Calls the eeg_data_SEstim2nifti function to convert EEG source estimate files to NIFTI format, including interpolation and thresholding, to facilitate comparison with other modalities.
    4. Handles any exceptions that occur during the process, logging the error details.
    """
    from utils.eeg_utils import eeg_data_stc2nifti
    
    # Log the start of the EEG data preparation
    print("\nStarting EEG data preparation...")
    logger.info("Starting EEG data preparation...")
    logger.info("  Detailed logs for this analysis will be saved to 'logger-eeg_stc2nifti.log'.")
    
    # Set up logging for EEG data preparation
    logger_eeg = setup_logging(LOCAL_DIR, 'logger-eeg_stc2nifti')
    
    try:
        # Process EEG data and convert to NIFTI format
        eeg_data_stc2nifti(LOCAL_DIR, logger_eeg, SUBJECTS, TASKS_CONDITIONS, EEG_TASKS_TP, THRESHOLD_PARAMS, EEG_STC_ALGO)
    except Exception as e:
        # Log any errors that occur during the data preparation
        logger.error(f"  An error occurred during `eeg_data_stc2nifti`: {e}")
        raise

def meg_data_preparation():
    """
    Initializes the MEG data preparation process.
    
    This function:
    1. Starts by logging the beginning of the MEG data preparation process.
    2. Sets up a logger to capture detailed logs specific to MEG data preparation.
    3. Calls the meg_data_SEstim2nifti function to convert MEG source estimate files to NIFTI format, including interpolation and thresholding, to facilitate comparison with other modalities.
    4. Handles any exceptions that occur during the process, logging the error details.
    """
    from utils.meg_utils import meg_data_stc2nifti

    # Log the start of the MEG data preparation
    print("\nStarting MEG data preparation...")
    logger.info("Starting MEG data preparation...")
    logger.info("  Detailed logs for this analysis will be saved to 'logger-meg_stc2nifti.log'.")

    # Setup logging
    logger_meg = setup_logging(LOCAL_DIR, 'logger-meg_stc2nifti')

    try:
        # Process EEG data and convert to NIFTI format
        meg_data_stc2nifti(LOCAL_DIR, logger_meg, SUBJECTS, TASKS_CONDITIONS, MEG_TASKS_TP, THRESHOLD_PARAMS, 
                           POS, MEG_STC_ALGO, mri_resolution = MRI_RESOLUTION, do_brain_masking = DO_BRAIN_MASKING, plotting = PLOTTING)
    except Exception as e:
        # Log any errors that occur during the data preparation
        logger.error(f"  An error occurred during `meg_data_stc2nifti`: {e}")
        raise

def generation_subs_rois():
    """
    Computes Regions of Interest (ROIs) for all specified subjects using a chosen brain atlas and generates ROIs in the 
    native space of each subject. This function iterates through each subject, applying the specified atlas to generate 
    ROIs, which are then transformed from the atlas template space to the subject's native space.

    This process involves:
    - Loading the subject's T1-weighted anatomical image.
    - Resampling the atlas to match the subject's T1 image.
    - Transforming the resampled atlas into the subject's native space.
    - Creating ROIs based on the atlas regions and saving them in the subject's directory.
    - Optionally plotting the ROIs if specified.

    The function relies on the `create_subject_rois` utility from the `roi_utils` module to perform the actual ROI 
    creation and transformation.
    """
    from utils.roi_utils import create_subject_rois

    # Log the start of the ROIs computation
    print(f"\nStarting ROIs computation from atlas {ATLAS}...")
    logger.info(f"Starting ROIs computation from atlas {ATLAS}...")
    
    # Start the time count to track the remaining computation time
    start_time = time.time()
    
    # Total number of subjects for progress tracking
    tot_perf = len(SUBJECTS)

    # Setup logging
    try:
        for count_perf, subject in enumerate(SUBJECTS):
            # Track the progress
            trackprogress(count_perf, tot_perf, start_time, logger)
            create_subject_rois(subject, ATLAS, LOCAL_DIR, PATH_TO_TPLFLOW, logger, plotting=PLOTTING)

    except Exception as e:
        logger.error(f"An error occured during `create_subject_rois`: {e}.")
        raise
    

def generation_comparison_modalities_df():
    """
    Perform comparison between EEG (resp. EEG) and fMRI modalities, including POA, COG distances, and overlap computations.

    This function initializes logging, processes each subject and task, and computes the necessary comparison metrics. 
    Results are stored in DataFrames for each subject and task.
    """

    from utils.df_utils import all_subject_POA_df, all_subject_COG_df, all_subject_overlap_df, subject_POA_df, subject_COG_df, subject_overlap_df
    from utils.generation_compa_df_utils import comparison_EEG_MEG_fMRI_df_fill

    # Setup logging
    print("\nStarting EEG (or resp. MEG) and fMRI comparison analysis, including POA, wCOG distances, and overlap computations.")
    logger.info("Starting EEG (or resp. MEG) and fMRI comparison analysis, including POA, wCOG distances, and overlap computations.")
    logger.info("  Detailed logs for this analysis will be saved to 'comparison_analysis.log'.")
    logger_comparison = setup_logging(LOCAL_DIR, 'logger-generation_df_comparison')

    # Initialize DataFrames to store results for all subjects
    df_all_sub = {'POA': all_subject_POA_df(), 'COG': all_subject_COG_df(), 'Overlap': all_subject_overlap_df()}

    # Start the time count to track the remaining computation time
    start_time = time.time()
    
    # Total number of subjects for progress tracking
    tot_perf = len(SUBJECTS)
    
    # Process each subject
    try:
        for count_perf, subject in enumerate(SUBJECTS, start=0):
            logger_comparison.info(f'Analyzing subject {subject}.')
            print(f'\nAnalyzing subject {subject}.')

            # Define the path to the subject derivatives directory
            sub_derivatives_outdir = Path(LOCAL_DIR) / 'derivatives' / subject

            # Track the progress
            trackprogress(count_perf, tot_perf, start_time, logger_comparison)
            
            # Initialize DataFrames for this subject
            df_sub = {'POA': subject_POA_df(), 'COG': subject_COG_df(), 'Overlap': subject_overlap_df()}
            
            # Process each task
            for task, condition in TASKS_CONDITIONS.items():
                logger_comparison.info(f"  Analyzing task {task}.")
                
                # Get correction parameters for this task
                corr_param = CorrectionParam(task)
                
                # Perform comparison
                comparison_EEG_MEG_fMRI_df_fill(sub_derivatives_outdir, subject, task, condition, df_all_sub, df_sub, corr_param, THRESHOLD_PARAMS, EEG_STC_ALGO, POS, logger_comparison, FMRI_COMPARISON_CLUSTERS, DO_BRAIN_MASKING, ROI_MASKING, ATLAS)
                
    except Exception as e:
        logger.error(f'  An error occured during `comparison_EEG_MEG_fMRI_acquisition`: {e}')
        raise (e)
    
    # Display results
    display(df_all_sub['POA'])
    display(df_all_sub['COG'])
    display(df_all_sub['Overlap'])

def mean_std_computation_step():
    """
    Computes the mean and standard deviation of distance metrics for each modality and analysis type.

    This function iterates over the specified analysis types ('POA' and 'COG') and computes
    the mean and standard deviation of distance metrics for various combinations of time points 
    (tp) and conditions using the `compute_mean_std` function. The results are logged and 
    exceptions are handled to ensure smooth execution.
    """
    from utils.stats_utils import compute_mean_std
    
    # Define analysis types
    analysis_types = ['POA', 'COG']

    # Compute mean and std for each analysis type
    for analysis_type in analysis_types:
        try:
            print(f'\nStarting {analysis_type} mean and std computation.')
            logger.info(f'Starting {analysis_type} mean and std computation.')
            
            # Compute mean and std
            compute_mean_std(LOCAL_DIR, analysis_type)
            
            print(f'\n{analysis_type} mean and std computation completed successfully.')
            logger.info(f'{analysis_type} mean and std computation completed successfully.')
        
        except Exception as e:
            logger.error(f"An error occurred during {analysis_type} mean and std computation: {e}")
            raise

def mean_stats_analysis():
    """
    Perform statistical analysis to determine if the mean of dEEG and dMEG differs from zero for both POA and COG analyses.
    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.
    Exception
        If an error occurs during the execution of the R script.
    """

    try:
        print('\nBeginning POA mean analysis.')
        logger.info('Beginning POA mean analysis.')
        
        csv_local_dir = Path(LOCAL_DIR) / 'derivatives' / 'results' / 'POA'
        if not csv_local_dir.exists():
            raise FileNotFoundError(f"The directory {csv_local_dir} does not exist.")
            
        mean_diff_zero(R_WORKING_DIRECTORY, str(csv_local_dir), RSCRIPT_EXECUTABLE, analysis_type='POA', nb_permu=NB_PERMUTATIONS, null_value=MEAN_COMPA_VALUE, score=MEAN_TYPE_ANALYSIS)
    except Exception as e:
        logger.error(f"An error occurred during the POA analysis: {e}")
        raise
    
    try:
        print('\nBeginning COG mean analysis.')
        logger.info('Beginning COG mean analysis.')
        
        csv_local_dir = Path(LOCAL_DIR) / 'derivatives' / 'results' / 'COG'
        if not csv_local_dir.exists():
            raise FileNotFoundError(f"The directory {csv_local_dir} does not exist.")
            
        mean_diff_zero(R_WORKING_DIRECTORY, str(csv_local_dir), RSCRIPT_EXECUTABLE, analysis_type='COG', nb_permu=NB_PERMUTATIONS, null_value=MEAN_COMPA_VALUE, score=MEAN_TYPE_ANALYSIS)
    except Exception as e:
        logger.error(f"An error occurred during the COG analysis: {e}")
        raise


def dispersion_stats_analysis():
    """
    Perform dispersion statistical analysis using either R or Python scripts.
    Raises
    ------
    ValueError
        If `DISPERSION_SCRIPT` is not 'R' or 'Python'.
    FileNotFoundError
        If the `xlsx_local_dir` does not exist.
    """
    if DISPERSION_SCRIPT not in ['R', 'Python']:
        raise ValueError("Only 'R' or 'Python' scripts are available.")
    
    logger.info(f"Using the {DISPERSION_SCRIPT} script for dispersion analysis.")
    print(f"Using the {DISPERSION_SCRIPT} script for dispersion analysis.")
    
    csv_local_dir_POA = Path(LOCAL_DIR) / 'derivatives' / 'results' / 'POA'
    if not csv_local_dir_POA.exists():
        raise FileNotFoundError(f"The directory {csv_local_dir_POA} does not exist.")

    csv_local_dir_COG = Path(LOCAL_DIR) / 'derivatives' / 'results' / 'COG'
    if not csv_local_dir_COG.exists():
        raise FileNotFoundError(f"The directory {csv_local_dir_COG} does not exist.")
    
    try:
        if DISPERSION_SCRIPT == 'R':
            print('\nBeginning POA dispersion analysis using R.')
            logger.info('Beginning POA dispersion analysis using R.')
            dispersion_analysis_R(R_WORKING_DIRECTORY, csv_local_dir_POA, RSCRIPT_EXECUTABLE, analysis_type='POA', nb_permu=NB_PERMUTATIONS, dist_metric=DISPERSION_DIST_METRIC)

            print('\nBeginning COG dispersion analysis using R.')
            logger.info('Beginning COG dispersion analysis using R.')
            dispersion_analysis_R(R_WORKING_DIRECTORY, csv_local_dir_COG, RSCRIPT_EXECUTABLE, analysis_type='COG', nb_permu=NB_PERMUTATIONS, dist_metric=DISPERSION_DIST_METRIC)
        
        elif DISPERSION_SCRIPT == 'Python':
            print('\nBeginning POA dispersion analysis using Python.')
            logger.info('Beginning POA dispersion analysis using Python.')
            dispersion_analysis_python(csv_local_dir_POA, analysis_type='POA', nb_permu=NB_PERMUTATIONS, dist_metric=DISPERSION_DIST_METRIC)

            print('\nBeginning COG dispersion analysis using Python.')
            logger.info('Beginning COG dispersion analysis using Python.')
            dispersion_analysis_python(csv_local_dir_COG, analysis_type='COG', nb_permu=NB_PERMUTATIONS, dist_metric=DISPERSION_DIST_METRIC)
    except Exception as e:
        logger.error(f"An error occurred during the dispersion analysis using {DISPERSION_SCRIPT}: {e}")
        raise


def main():
    # Print the configuration and ask for confirmation
    print_configuration(logger)

    print(f'\nThe subjects studied are: {SUBJECTS}')

    sections = [
        (fmri_data_preparation,  "Preparation of fMRI Data."),
        (eeg_data_preparation,  "Prepares EEG data."),
        (meg_data_preparation,  "Prepares MEG data."),
        (generation_subs_rois, "Generates Regions of Interest (ROIs) in the subject's native space."),
        (generation_comparison_modalities_df,  "Generation of DataFrames to compares EEG and MEG modalities to fMRI."),
        (mean_std_computation_step, "Calculation of the mean and standard deviation for POA and COG analyses, covering both EEG and MEG modalities."),
        (mean_stats_analysis, "Compares dMEG and dEEG means to zero."),
        (dispersion_stats_analysis, "Compares variability between dEEG and dMEG.")
    ]

    # Ask the user if they want to run all sections or select specific ones
    print("\nAvailable sections to run:")
    for i, (_, section_name) in enumerate(sections):
        print(f"{i + 1}. {section_name}")
        
    while True:
        run_all = input("\nDo you want to run all sections? (yes/no): ").strip().lower()
        if run_all in ['yes', 'no']:
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    
    if run_all == 'yes':
        # Run all sections
        for section_function, section_name in sections:
            section_function()

    else:
        # Allow user to choose sections to run
        choose_sections_to_run(sections)

    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()