"""
stats_utils.py

This script performs various statistical and dispersion analyses on EEG and MEG neuroimaging data using both R and Python.

Functions:
1. `mean_diff_zero`: Executes an R script (`diff_zero.R`) to analyze whether the mean difference of dEEG and dMEG is zero. Results are saved in Excel with the name format "all_subjects_analysis-{analysis_type}_modality_comparison_analysis-location_R.csv".
   - Parameters include the working directory for the R script, local directory for data, Rscript executable path, type of analysis (`POA` or `COG`), number of permutations, null value for comparison, and scoring method (`sign` or `rank`).

2. `dispersion_analysis_R`: Runs an R script (`dispersion_analysis.R`) to compare the dispersion of dEEG and dMEG data. Results are saved in Excel with the name format "all_subjects_analysis-{analysis_type}_modality_comparison_analysis-dispersion_R.csv".
   - Parameters include the working directory for the R script, local directory for data, Rscript executable path, type of analysis (`POA` or `COG`), number of permutations (capped at 999), and distance metric (`euclidean`, `manhattan`, etc.).

3. `dispersion_analysis_python`: Performs dispersion analysis directly in Python, using data from an Excel file. It reshapes the data, computes dispersion metrics, and saves results in Excel with the name format "all_subjects_analysis-{analysis_type}_modality_comparison_analysis-dispersion.csv".
   - Parameters include the local directory for data, type of analysis (`POA` or `COG`), number of permutations (capped at 999), and distance metric.

4. `compute_mean_std(local_dir, analysis_type)`: Computes the mean and standard deviation of distance metrics for EEG and MEG modalities. Results are saved in Excel with the name format "all_subjects_analysis-{analysis_type}_modality-{modality}_stats-MeanStd.csv".
"""


from pathlib import Path
import subprocess
import warnings

from utils.utils import load_df_from_csv, flatten_columns
from utils.dispersion_stats_analysis_utils import *

def mean_diff_zero(working_directory, local_dir, rscript_executable, analysis_type='POA', nb_permu=1000, null_value=None, score='sign'):
    '''
    This function uses the R script defined in `diff_zero.R` to analyze whether the mean of dEEG and dMEG differs from zero. 
    The results are saved in `local_dir` under the name: "all_subjects_analysis-{analysis_type}_modality_comparison_analysis-location_R.xlsx".
    
    Parameters
    ----------
    working_directory : str
        The directory where the R script `diff_zero.R` is located.
    local_dir : str
        The directory where the output results should be saved and from where the data are read.
    rscript_executable : str
        The path to the `Rscript` executable.
    analysis_type : str, optional, default='POA'
        Specifies the type of analysis to perform. Must be either 'POA' or 'COG'.
    nb_permu : int, optional, default=1000
        The number of permutations for the analysis. Will be capped at 999 if exceeded.
    null_value : float or None, optional, default=None
        The value to which the variables are compared.
    score : str, optional, default='sign'
        The scoring method to be used in the analysis. Must be either 'sign' or 'other'.
    
    Raises
    ------
    ValueError
        If `analysis_type` is not 'POA' or 'COG', or if `score` is not 'sign' or 'other'.
    FileNotFoundError
        If the specified R script cannot be found at the provided `working_directory`.
    subprocess.CalledProcessError
        If the R script execution fails.
    '''
    print(f'Beginning statistical analysis to determine if the mean of dEEG and dMEG differs from zero for the {analysis_type} analysis.')
    
    # Validate the analysis type
    if analysis_type not in ['POA', 'COG']:
        raise ValueError("The `analysis_type` parameter should be either 'POA' or 'COG'.")
    
    # Validate the scoring method
    if score not in ['sign', 'rank']:
        raise ValueError("The `score` parameter should be either 'sign' or 'rank'.")
    
    # Define the path to the R script
    location_R_script = Path(working_directory) / 'diff_zero.R'
    print(f"R script path: {location_R_script}")
    
    # Ensure the R script exists
    if not location_R_script.exists():
        raise FileNotFoundError(f"The R script at {location_R_script} does not exist.")
    
    # Construct the command to execute the R script
    command = [
        rscript_executable, 
        str(location_R_script), 
        f'working_directory={working_directory}', 
        f'path_to_data={local_dir}', 
        f'analysis_type={analysis_type}',
        f'nb_permu={nb_permu}',
        f'null_value={"NULL" if null_value is None else null_value}',
        f'score={score}'
    ]
    
    # Print command for debugging
    print(f"Executing command: {' '.join(command)}")
    
    # Execute the R script
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("R script output:", result.stdout)
        print("R script errors (if any):", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the R script: {e}")
        print(f"R script output: {e.output}")
        print(f"R script errors: {e.stderr}")
        raise


def dispersion_analysis_R(working_directory, local_dir, rscript_executable, analysis_type='POA', nb_permu=999, dist_metric='euclidean'):
    '''
    This function uses the R script defined in `dispersion_analysis.R` to analyze whether the dispersion of dEEG and dMEG differs from each other. 
    The results are saved in `local_dir` under the name: "all_subjects_analysis-{analysis_type}_modality_comparison_analysis-dispersion_R.xlsx".
    
    Parameters
    ----------
    working_directory : str
        The directory where the R script `dispersion_analysis.R` is located.
    local_dir : str
        The directory where the output results should be saved and from where the data are read.
    rscript_executable : str
        The path to the `Rscript` executable.
    analysis_type : str, optional, default='POA'
        Specifies the type of analysis to perform. Must be either 'POA' or 'COG'.
    nb_permu : int, optional, default=999
        The number of permutations for the dispersion analysis.
    dist_metric : str, optional, default='euclidean'
        The distance metric used in the dispersion analysis. E.g., 'euclidean', 'manhattan'.
    
    Raises
    ------
    ValueError
        If `analysis_type` is not 'POA' or 'COG'.
    FileNotFoundError
        If the specified R script cannot be found at the provided `working_directory`.
    subprocess.CalledProcessError
        If the R script execution fails.
    '''
    
    print(f'Beginning dispersion analysis to determine if the dispersion of dEEG and dMEG differs from each other for the {analysis_type}.')
    
    # Validate the analysis type
    if analysis_type not in ['POA', 'COG']:
        raise ValueError("The `analysis_type` parameter should be either 'POA' or 'COG'.")

    if nb_permu > 999:
        warnings.warn('The maximum number of permutations is 999. The value of `nb_permu` has been capped at 999.', UserWarning)
        nb_permu = 999
    
    # Define the path to the R script
    location_R_script = Path(working_directory) / 'dispersion_analysis.R'
    print(f"R script path: {location_R_script}")
    
    # Ensure the R script exists
    if not location_R_script.exists():
        raise FileNotFoundError(f"The R script at {location_R_script} does not exist.")
    
    # Construct the command to execute the R script
    command = [
        rscript_executable, 
        str(location_R_script), 
        f'working_directory={working_directory}', 
        f'path_to_data={local_dir}', 
        f'analysis_type={analysis_type}', 
        f'nb_permu={nb_permu}', 
        f'dist_metric={dist_metric}'
    ]
    
    # Print command for debugging
    print(f"Executing command: {' '.join(command)}")
    
    # Execute the R script
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("R script output:", result.stdout)
        print("R script errors (if any):", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the R script: {e}")
        print(f"R script output: {e.output}")
        print(f"R script errors: {e.stderr}")
        raise
        
def dispersion_analysis_python(local_dir, analysis_type='POA', nb_permu=999, dist_metric='euclidean'):
    """
    Performs dispersion analysis on EEG and MEG data and saves the results to an Excel file.

    This function loads data from an Excel file, reshapes the data, computes dispersion analysis, 
    and saves the results to a specified Excel file.

    Parameters
    ----------
    local_dir : str
        Directory where the input Excel file is located and where the output results will be saved.
    analysis_type : str, optional, default='POA'
        Specifies the type of analysis to perform. Must be either 'POA' or 'COG'.
    nb_permu : int, optional, default=999
        Number of permutations for the dispersion analysis. The value is capped at 999 if provided value is higher.
    dist_metric : str, optional, default='euclidean'
        Distance metric to use for the dispersion analysis.

    Raises
    ------
    ValueError
        If `analysis_type` is not 'POA' or 'COG'.
    FileNotFoundError
        If the input Excel file does not exist.
    """
    
    # Validate parameters
    if analysis_type not in ['POA', 'COG']:
        raise ValueError("`analysis_type` should be either 'POA' or 'COG'.")

    # Define the path to the Excel file
    data_file_path = Path(local_dir) / f"all_subjects_analysis-{analysis_type}_modality_comparison.csv"
    
    # Check if the file exists
    if not data_file_path.exists():
        raise FileNotFoundError(f"The Excel file at {data_file_path} does not exist.")
    
    # Load DataFrame from the Excel file
    data_df = load_df_from_csv(data_file_path)
    
    # Flatten MultiIndex columns if necessary
    data_df.columns = flatten_columns(data_df.columns)
    
    # Optionally display the DataFrame to verify its contents
    # print(data_df.head())
    
    # Reshape DataFrame to combine MEG and EEG modalities with a modality column
    combined_data_df = reshape_dataframe(data_df)
    
    # Optionally display the reshaped DataFrame to verify its contents
    # print(combined_data_df.head())
    
    # Compute the dispersion analysis
    results_dispersion_analysis = compute_dispersion_analysis(
        combined_data_df, 
        analysis_type=analysis_type, 
        dist_metric=dist_metric, 
        nb_permutation=nb_permu,
        path_to_data = local_dir
    )
    
    # Display the results
    # print(results_dispersion_analysis.head())
    
    # Define the path for saving the dispersion results
    path_dispersion_results = Path(local_dir) / f"all_subjects_analysis-{analysis_type}_modality_comparison_stats-dispersion_python.csv"
    
    # Save the results to an Excel file
    results_dispersion_analysis.to_csv(path_dispersion_results, index=False)


def compute_mean_std(local_dir, analysis_type):
    """
    Computes the mean and standard deviation of distance metrics for EEG and MEG modalities.

    This function processes data from an Excel file containing modality comparisons and calculates 
    the mean and standard deviation of distance metrics for various conditions and time points. 
    The results are saved to a new Excel file for each modality and analysis type.

    Parameters
    ----------
    local_dir : str
        The base directory where the results and data files are located.
    
    analysis_type : str
        Specifies the type of analysis ('COG' or 'POA') which determines the specific folder and file to be used.

    This function performs the following steps:
    1. **Define Paths**: Constructs the paths for loading the input data and saving the output results.
    2. **Load and Prepare Data**: Loads the data from the specified Excel file, flattens column names, and reshapes the DataFrame for analysis.
    3. **Initialize Results DataFrame**: Prepares an empty DataFrame to store computed mean and standard deviation values.
    4. **Compute Statistics**:
        - Filters data by modality ('meg' or 'eeg').
        - Computes mean and standard deviation for all time points and conditions.
        - Computes mean and standard deviation separately for each time point and condition.
    5. **Save Results**: Writes the computed statistics to an Excel file specific to each modality and analysis type.

    Returns
    -------
    None
        The function saves the computed statistics directly to Excel files and does not return any value.

    Raises
    ------
    ValueError
        If `analysis_type` is not 'POA' or 'COG'.
    """

    # Validate parameters
    if analysis_type not in ['POA', 'COG']:
        raise ValueError("`analysis_type` should be either 'POA' or 'COG'.")

    # Create an empty DataFrame with the specified columns
    mean_std_df = pd.DataFrame(columns=[
        'modality', 'conditions', 'tp',
        'dist_x_mean', 'dist_x_std',
        'dist_y_mean', 'dist_y_std',
        'dist_z_mean', 'dist_z_std',
        'dist_mean', 'dist_std'
    ])

    def fill_mean_std_df_(df, modality, tp, condition):
        """
        Adds a new row to the mean and standard deviation DataFrame based on input data.

        This function calculates the mean and standard deviation of distance metrics for a given modality,
        time point (tp), and condition, and appends these statistics to the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing distance metrics with columns 'Dist_x', 'Dist_y', 'Dist_z', and 'Dist_norm'.
            
        modality : str
            The modality of the data (e.g., 'eeg', 'meg').
    
        tp : str
            The time point of the data (e.g., 'pre', 'post', or a numeric index).
    
        condition : str
            The experimental condition (e.g., 'rest', 'task').

        Returns
        -------
        None
            The function modifies the `mean_std_df` DataFrame in place by appending a new row.
        """
    
        nonlocal mean_std_df
        
        # Calculate statistics
        stats = {
            'modality': modality,
            'conditions': condition,
            'tp': tp,
            'dist_x_mean': np.round(np.nanmean(df['Dist_x']), 2),
            'dist_x_std': np.round(np.nanstd(df['Dist_x']), 2),
            'dist_y_mean': np.round(np.nanmean(df['Dist_y']), 2),
            'dist_y_std': np.round(np.nanstd(df['Dist_y']), 2),
            'dist_z_mean': np.round(np.nanmean(df['Dist_z']), 2),
            'dist_z_std': np.round(np.nanstd(df['Dist_z']), 2),
            'dist_mean': np.round(np.nanmean(df['Dist_norm']), 2),
            'dist_std': np.round(np.nanstd(df['Dist_norm']), 2)
        }
        
        # Create a DataFrame for the new row
        new_row = pd.DataFrame([stats])
    
        # Append the new row to the existing DataFrame
        if mean_std_df.empty:
            mean_std_df = new_row
        else:
            mean_std_df = pd.concat([mean_std_df, new_row], ignore_index=True)
        
    # Define paths for output directories and load the DataFrame
    df_outdir = Path(local_dir) / 'derivatives' / 'results' / analysis_type
    csv_path = df_outdir / f'all_subjects_analysis-{analysis_type}_modality_comparison.csv'
    df_ = load_df_from_csv(csv_path)
    
    # Flatten column names and reshape the DataFrame
    df_.columns = flatten_columns(df_.columns)
    reshaped_df_ = reshape_dataframe(df_)
    
    # Initialize results DataFrame for each modality
    for modality in ['meg', 'eeg']:
        # Filter the data for the current modality
        reshaped_df_modality = reshaped_df_[reshaped_df_['modality'] == modality]
        
        # Compute mean and std for 'all' time points and conditions
        fill_mean_std_df_(reshaped_df_modality, modality, 'all', 'all')
        
        # Compute mean and std for each time point
        for tp in range(3):
            reshaped_df_modality_tp = reshaped_df_modality[reshaped_df_modality['Info_tpindex'] == tp]
            fill_mean_std_df_(reshaped_df_modality_tp, modality, tp, 'all')
        
        # Compute mean and std for each condition
        for condition in np.unique(reshaped_df_modality['Info_condition']):
            reshaped_df_modality_condition = reshaped_df_modality[reshaped_df_modality['Info_condition'] == condition]
            fill_mean_std_df_(reshaped_df_modality_condition, modality, 'all', condition)
            
            # Compute mean and std for each time point within each condition
            for tp in range(3):
                reshaped_df_modality_condition_tp = reshaped_df_modality_condition[reshaped_df_modality_condition['Info_tpindex'] == tp]
                fill_mean_std_df_(reshaped_df_modality_condition_tp, modality, tp, condition)
        
        # Save the results to an Excel file
        mean_std_df_path = df_outdir / f'all_subjects_analysis-{analysis_type}_modality-{modality}_stats-MeanStd_python.csv'
        mean_std_df.to_csv(mean_std_df_path, index=False)