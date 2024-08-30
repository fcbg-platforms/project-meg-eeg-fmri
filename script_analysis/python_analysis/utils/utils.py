""" 
utils.py

This module contains utility functions designed to streamline various tasks, including logging setup, file management, and DataFrame operations. 

Functions:

- `setup_logging(local_dir, logging_name)`
  Configures logging to a file located in the specified directory.

- `track_progress(count_perf, tot_perf, start_time, logger)`  
  Displays processing progress, including estimated time remaining.

- `flatten_columns(columns)`  
  Converts MultiIndex columns into a single level by concatenating names.

- `load_df_from_csv(file_path)`  
  Loads a DataFrame from an csv file and reconstructs MultiIndex columns from flattened names.

- `save_df_to_csv(df, file_path)`
  Saves a DataFrame with MultiIndex columns to an Excel file.

- `found_condition_file(pattern)` 
  Searches for files in the filesystem that match a specified pattern.
""" 

import logging
import os
import re
from pathlib import Path
import pandas as pd
import glob
import time
import numpy as np

def setup_logging(local_dir, logging_name):
    """
    Set up logging to a file in a specified directory with a unique logger name.

    Parameters
    ----------
    local_dir : str
        The base directory where the logging directory will be created.
    logging_name : str
        The name of the logging file (without extension).

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    # Path to the log file
    logging_outdir = Path(local_dir) / 'derivatives' / 'logging'
    logging_outdir.mkdir(parents=True, exist_ok=True)
    log_file = logging_outdir / f'{logging_name}.log'
    
    # Create a logger
    logger = logging.getLogger(logging_name)
    logger.setLevel(logging.INFO)  # Set default logging level

    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler for the logger
    file_handler = logging.FileHandler(log_file, mode='w')  # Overwrite mode
    file_handler.setLevel(logging.INFO)  # Set the logging level for this handler

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def trackprogress(count_perf, tot_perf, start_time, logger):
    """
    Prints the progress of processing, including estimated remaining time.
    
    Parameters:
    - count_perf: The index of the current subject in the list.
    - tot_perf: The total number of subjects to process.
    - start_time: The start time of the processing to estimate the remaining time.
    - logger: The logging object used for logging progress.
    """   
    # Track the progress
    progress = f'{count_perf / tot_perf * 100:.2f} % completed'
    logger.info(progress)
    print(progress)
    
    if count_perf > 0:
        elapsed_time = time.time() - start_time
        estimated_time_remaining = (elapsed_time / count_perf) * (tot_perf - count_perf)
        hours_remaining = np.floor(estimated_time_remaining / 3600)
        minutes_remaining = np.floor((estimated_time_remaining % 3600) / 60)
        remaining_time = f'Estimated remaining time: {hours_remaining} hours, {minutes_remaining} minutes.'
        logger.info(remaining_time)
        print(remaining_time)

def flatten_columns(columns):
    """
    Convert MultiIndex columns into a single level of columns with concatenated names.

    Parameters:
    - columns (pd.MultiIndex): The MultiIndex object representing the columns of a DataFrame.

    Returns:
    - List[str]: A list of column names where MultiIndex tuples are joined into single strings, separated by underscores.

    Example:
    - If `columns` is a MultiIndex with levels [('A', 'B'), ('C', 'D')], the output will be ['A_C', 'A_D', 'B_C', 'B_D'].
    """
    flattened = ['_'.join(map(str, col)).strip() for col in columns.to_flat_index()]
    return flattened

def load_df_from_csv(file_path):
    """
    Load a DataFrame from a CSV file, reconstructing MultiIndex columns from flattened column names.

    Parameters:
    - file_path (str): Path to the CSV file containing the DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with columns reconstructed into a MultiIndex format.

    Details:
    - The function reads the CSV file into a flat DataFrame, then reconstructs MultiIndex columns by splitting 
      the concatenated column names, which were flattened into single-level names with underscores.

    Example:
    - If the CSV file has columns named 'A_B' and 'C_D', they will be reconstructed into a MultiIndex with 
      levels ['A', 'C'] and ['B', 'D'].
    """
    df_flat = pd.read_csv(file_path)
    
    # Reconstruct MultiIndex columns
    cols = pd.MultiIndex.from_tuples([tuple(col.split('_')) for col in df_flat.columns])
    df = pd.DataFrame(df_flat.values, columns=cols)
    
    return df

def save_df_to_csv(df, file_path):
    """
    Save a DataFrame with MultiIndex columns to a CSV file.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with MultiIndex columns to be saved.
    - file_path (str or Path): Path where the CSV file will be saved.
    """
    # Convert all parts of the MultiIndex columns to strings
    tuples = [tuple(map(str, col)) for col in df.columns.to_flat_index()]
    df.columns = pd.MultiIndex.from_tuples(tuples)

    # Flatten the MultiIndex columns into a single level
    df_flat = pd.DataFrame(df.to_records(index=False))
    df_flat.columns = ['_'.join(col).strip() for col in df.columns.to_flat_index()]

    # Save the flattened DataFrame to CSV
    try:
        df_flat.to_csv(file_path, index=False)
        # print(f"Successfully saved DataFrame to {file_path}")
    except Exception as e:
        raise ValueError(f"Error saving DataFrame to {file_path}: {e}")


def found_condition_file(pattern):
    """
    Searches for files matching a specified pattern in the filesystem.

    Parameters:
    - pattern (str): The file pattern to match. This can include wildcard characters like '*' or '?'.
    Returns:
    - List[str]: A list of file paths that match the specified pattern.
    Raises:
    - FileNotFoundError: If no files match the specified pattern.
    """
    
    matching_files = list(glob.glob(str(pattern)))

    if len(matching_files) == 0:
        raise FileNotFoundError(f"No matching files found for pattern: {pattern}.")

    return matching_files






