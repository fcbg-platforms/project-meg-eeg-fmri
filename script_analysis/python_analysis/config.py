# Configuration File

# Base directory for the BIDS-compliant dataset
LOCAL_DIR: str = "I:\\9008_CBT_HNP\\MEG-EEG-fMRI\\dataset_bids"

# Directory containing the raw fMRI data for conversion to BIDS format
PATH_TO_MRI_DATA: str = "I:\\9008_CBT_HNP\\MEG-EEG-fMRI\\MRI_analyses"

# Directory containing the Excel files for fMRI data conversion information (e.g., Corresp_SPM_con_x.xlsx files)
MRI_DATA_CONV_INFO_DIR: str = "C:\\Users\\lilia.ponselle\\Documents\\GitHub\\Internship_MEEG_IRM\\data_info"

# Directory path to TemplateFlow's MNI152NLin6Asym template
PATH_TO_TPLFLOW: str = "C:\\Users\\lilia.ponselle\\.cache\\templateflow\\tpl-MNI152NLin6Asym"

# Working directory for R scripts and analyses
R_WORKING_DIRECTORY: str = "C:\\Users\\lilia.ponselle\\Documents\\GitHub\\Internship_MEEG_IRM\\r_scripts"

# Path to the Rscript executable for running R scripts
RSCRIPT_EXECUTABLE: str = "C:\\Program Files\\R\\R-4.4.1\\bin\\Rscript.exe"

# Create a list of subjects from 'sub-03' to 'sub-17' excluding 'sub-10' and 'sub-11'
SUBJECTS: list = [f"sub-{i:02}" for i in range(3, 18) if i not in [10, 11]]

# Define thresholding parameters
THRESHOLD_PARAMS: dict = {
    'EEG': {'alpha': 0.01, 'cluster_thresh': 15},
    'MEG': {'alpha': 0.01, 'cluster_thresh': 15}
}

# For each task associate the contrast of interest
TASKS_CONDITIONS: dict ={
    "aud": ['sine', 'piano1', 'piano4'],
    "feet": ['leftfoot', 'rightfoot'],
    "fing": ['finger'],
    "vis": ['face','scramble']
}

# Define time point windows for each task, which will be converted to NIFTI format for comparison with other modalities
# In frames
EEG_TASKS_TP: dict = {
    "aud": [[170,205], [210,260], [260,301]], 

    "feet": {'leftfoot': [[180, 205], [205, 251], [275, 326]],
             'rightfoot': [[140,191], [237,255], [255,286]] },
    "fing": [[200,235], [235, 285], [285, 351]],
    "vis": [[140,190], [190,230], [230,271]]
}

# In s
MEG_TASKS_TP: dict = {
    'aud': [[0.160, 0.200],[0.200, 0.260],[0.260, 0.320]],
    'feet': [[0.260, 0.350],[0.450, 0.530],[0.600, 0.700]],
    'fing': [[0.060, 0.170],[0.170, 0.260],[0.260, 0.400]],
    'vis': [[0.070, 0.110],[0.110, 0.180],[0.250, 0.400]]
}

# Parameters for EEG Data Conversion and Processing
EEG_STC_ALGO: str = 'LAURA'

# Parameters for MEG Data Conversion and Processing
POS: int = 3  # Spatial resolution of the source estimates, specified in millimeters (3 mm or 5 mm).
DO_BRAIN_MASKING: bool = False  # Whether to apply a brain mask to exclude non-brain regions from the data. Set to True to enable masking.
MRI_RESOLUTION: bool = False  # Whether to resample the output images to match MRI resolution. Set to True for high-resolution output.
MEG_STC_ALGO: str = 'sLORETA'

# Visualization Settings
PLOTTING: bool = False  # Enable or disable the plotting of thresholded MEG, EEG, and fMRI results. Set to True to visualize the data.

# Comparison DataFrame Generation Parameters
FMRI_COMPARISON_CLUSTERS: list = [1, 2, 3]
ROI_MASKING: bool = False

# Statistical Analysis parameters
NB_PERMUTATIONS: int = 999
DISPERSION_DIST_METRIC: str = 'euclidean'
MEAN_COMPA_VALUE: list = None # Should stay None if want to compare to 0
MEAN_TYPE_ANALYSIS: str = 'sign'
DISPERSION_SCRIPT: str = 'R'  # 'R' or 'Python'

# Rois generation parameters
ATLAS = 'HOCPALth0'

## Def the Class  for MRI correction param
class CorrectionParam:
    """
    A class to encapsulate correction parameters for fMRI data analysis.
    
    Attributes:
        fMRI_ALPHA (float): The alpha level for statistical significance.
        fMRI_CORRECTION_METHOD (str): The method used for correction (e.g., 'fpr' or 'fdr').
        fMRI_CLUSTER_THRES (int): The threshold for cluster size.
        fMRI_TWOSIDED (bool): Indicates whether a two-sided test is used.
    """
    
    # Dictionary of correction settings for different tasks
    CORRECTION_SETTINGS = {
        'aud': {'fMRI_ALPHA': 0.01, 'fMRI_CORRECTION_METHOD': 'fpr', 'fMRI_CLUSTER_THRES': 10, 'fMRI_TWOSIDED': False},
        'aud_all': {'fMRI_ALPHA': 0.01, 'fMRI_CORRECTION_METHOD': 'fpr', 'fMRI_CLUSTER_THRES': 10, 'fMRI_TWOSIDED': False},
        'default': {'fMRI_ALPHA': 0.01, 'fMRI_CORRECTION_METHOD': 'fdr', 'fMRI_CLUSTER_THRES': 10, 'fMRI_TWOSIDED': False}
    }

    def __init__(self, task: str):
        """
        Initializes the CorrectionParam instance with parameters specific to the given task.
        
        Args:
            task (str): The task name to fetch the corresponding correction settings.
        
        """
        settings = self.CORRECTION_SETTINGS.get(task, self.CORRECTION_SETTINGS['default'])
        
        # Set instance attributes based on settings
        self.fMRI_ALPHA = settings['fMRI_ALPHA']
        self.fMRI_CORRECTION_METHOD = settings['fMRI_CORRECTION_METHOD']
        self.fMRI_CLUSTER_THRES = settings['fMRI_CLUSTER_THRES']
        self.fMRI_TWOSIDED = settings['fMRI_TWOSIDED']


def print_configuration(logger):
    """
    Print the current configuration and ask for user approval.
    """
    def print_params(param_name, params):
        print(f"\n{param_name} thresholding parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

    def log_params(param_name, params):
        logger.info(f"{param_name} thresholding parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")

    print("\nCurrent Configuration:")
    print('\nGeneral Settings:') 
    print(f"  LOCAL_DIR: {LOCAL_DIR}")
    print(f"  PATH_TO_MRI_DATA: {PATH_TO_MRI_DATA}")
    print(f"  MRI_DATA_CONV_INFO_DIR: {MRI_DATA_CONV_INFO_DIR}")
    print(f"  PATH_TO_TPLFLOW: {PATH_TO_TPLFLOW}")
    print(f"  R_WORKING_DIRECTORY: {R_WORKING_DIRECTORY}")
    print(f"  RSCRIPT_EXECUTABLE: {RSCRIPT_EXECUTABLE}")
    print(f"  SUBJECTS: {SUBJECTS}")

    logger.info("Current Configuration:")
    logger.info("General Settings:")
    logger.info(f"  LOCAL_DIR: {LOCAL_DIR}")
    logger.info(f"  PATH_TO_MRI_DATA: {PATH_TO_MRI_DATA}")
    logger.info(f"  MRI_DATA_CONV_INFO_DIR: {MRI_DATA_CONV_INFO_DIR}")
    logger.info(f"  R_WORKING_DIRECTORY: {R_WORKING_DIRECTORY}")
    logger.info(f"  RSCRIPT_EXECUTABLE: {RSCRIPT_EXECUTABLE}")
    logger.info(f"  SUBJECTS: {SUBJECTS}")
    
    print(f"\n  TASKS_CONDITIONS:")
    logger.info(f"\n  TASKS_CONDITIONS:")
    for task, condi in TASKS_CONDITIONS.items():
        print(f"    Task {task}: {condi}")
        logger.info(f"    Task {task}: {condi}")
    
    print(f"\n  EEG_TASKS_TP:")
    logger.info(f"\n  EEG_TASKS_TP:")
    for task, tp_window in EEG_TASKS_TP.items():
        print(f"    Task {task}: {tp_window}")
        logger.info(f"    Task {task}: {tp_window}")

    print(f"\n  MEG_TASKS_TP:")
    logger.info(f"\n  MEG_TASKS_TP:")
    for task, tp_window in MEG_TASKS_TP.items():
        print(f"    Task {task}: {tp_window}")
        logger.info(f"    Task {task}: {tp_window}")

    print("\nVisualization Settings:")
    print(f"  PLOTTING: {PLOTTING}")
    logger.info("Visualization Settings:")
    logger.info(f"  PLOTTING: {PLOTTING}")

    print("\nEEG Processing Parameters:")
    print(f"  EEG_STC_ALGO: {EEG_STC_ALGO}")

    logger.info("nEEG Processing Parameters:")
    logger.info(f"  EEG_STC_ALGO: {EEG_STC_ALGO}")

    print("\nMEG Processing Parameters:")
    print(f"  POS: {POS}")
    print(f"  DO_BRAIN_MASKING: {DO_BRAIN_MASKING}")
    print(f"  MRI_RESOLUTION: {MRI_RESOLUTION}")
    print(f"  MEG_STC_ALGO: {MEG_STC_ALGO}")

    logger.info("MEG Processing Parameters:")
    logger.info(f"  POS: {POS}")
    logger.info(f"  DO_BRAIN_MASKING: {DO_BRAIN_MASKING}")
    logger.info(f"  MRI_RESOLUTION: {MRI_RESOLUTION}")
    logger.info(f"  MEG_STC_ALGO: {MEG_STC_ALGO}")

    # Print EEG and MEG thresholding parameters
    for param_name, params in THRESHOLD_PARAMS.items():
        print_params(param_name, params)
        log_params(param_name, params)

    # Print and log fMRI correction parameters
    for task in ['aud', 'other']:
        correction_params = CorrectionParam(task)
        print(f'\nFor {task} tasks, the fMRI correction parameters are:')
        print(f"  fMRI_ALPHA: {correction_params.fMRI_ALPHA}")
        print(f"  fMRI_CORRECTION_METHOD: {correction_params.fMRI_CORRECTION_METHOD}")
        print(f"  fMRI_CLUSTER_THRES: {correction_params.fMRI_CLUSTER_THRES}")
        print(f"  fMRI_TWOSIDED: {correction_params.fMRI_TWOSIDED}")

        logger.info(f"For {task} tasks, the fMRI correction parameters are:")
        logger.info(f"  fMRI_ALPHA: {correction_params.fMRI_ALPHA}")
        logger.info(f"  fMRI_CORRECTION_METHOD: {correction_params.fMRI_CORRECTION_METHOD}")
        logger.info(f"  fMRI_CLUSTER_THRES: {correction_params.fMRI_CLUSTER_THRES}")
        logger.info(f"  fMRI_TWOSIDED: {correction_params.fMRI_TWOSIDED}")

    print('\nROIs Generation Parameters:')
    print(f"  ATLAS: {ATLAS}")

    logger.info('\nROIs Generation Parameters:')
    logger.info(f"  ATLAS: {ATLAS}")

    print("\nComparison df Generation Parameters:")
    print(f"  FMRI_COMPARISON_CLUSTERS: {FMRI_COMPARISON_CLUSTERS}")
    print(f"  ROI_MASKING: {ROI_MASKING}")

    logger.info("Comparison df Generation Parameters:")
    logger.info(f"  FMRI_COMPARISON_CLUSTERS: {FMRI_COMPARISON_CLUSTERS}")
    
    print('\nStatistical Analysis Parameters:')
    print(f"  NB_PERMUTATIONS: {NB_PERMUTATIONS}")
    print(f"  DISPERSION_SCRIPT: {DISPERSION_SCRIPT}")
    print(f"  DISPERSION_DIST_METRIC: {DISPERSION_DIST_METRIC}")
    print(f"  MEAN_COMPA_VALUE: {MEAN_COMPA_VALUE}")
    print(f"  MEAN_TYPE_ANALYSIS: {MEAN_TYPE_ANALYSIS} test")

    logger.info('Statistical Analysis Parameters:')
    logger.info(f"  NB_PERMUTATIONS: {NB_PERMUTATIONS}")
    logger.info(f"  DISPERSION_SCRIPT: {DISPERSION_SCRIPT}")
    logger.info(f"  DISPERSION_DIST_METRIC: {DISPERSION_DIST_METRIC}")
    logger.info(f"  MEAN_COMPA_VALUE: {MEAN_COMPA_VALUE}")
    logger.info(f"  MEAN_TYPE_ANALYSIS: {MEAN_TYPE_ANALYSIS} test")

    # Ask user to confirm configuration
    while True:
        proceed = input("\nAre these settings correct? (yes/no): ").strip().lower()
        if proceed in ['yes', 'no']:
            if proceed == 'no':
                print("Please update the variables in the config.py file and rerun.")
                logger.info("User did not confirm configuration. Exiting script.")
                sys.exit()  # Exit the script
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")