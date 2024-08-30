"""window_utils.py

Module with function to handle the tp selction in defined window for eeg and meg stc to nifti transformation

Functions:
- get_max_index(values): Returns the index of the maximum value in a 2D array.
- get_poa_timepoint_sestim(values_window, window): Computes the source estimate for the 
  time point with the maximum value in the time course.
- get_poa_window_sestim(values_window, window): Computes the mean source estimate around 
  the time point with the maximum value in the time course.
- compute_window_indices(window, t_start, t_end, t_step, stc_length): Compute the start and end indices 
  for a time window, handling out-of-bounds cases.

"""
import numpy as np
import warnings

def get_max_index(values):
    """
    Returns the index of the maximum value in the 2D array.
    
    Parameters:
    values (np.ndarray): A 2D numpy array.
    
    Returns:
    tuple: A tuple of indices representing the position of the maximum value.
    """
    return np.unravel_index(np.argmax(values, axis=None), values.shape)

def get_poa_timepoint_sestim(values_window, window):
    """
    Computes the source estimate for the time point with the maximum value in the time course.
    
    Parameters:
    values_window (np.ndarray): A 2D numpy array where rows represent different sources and columns represent different time points.
    window (np.ndarray): A 1D numpy array representing the time window.
    
    Returns:
    tuple: A tuple containing:
        - sestim_tp_values (np.ndarray): Source estimate values at the time point of maximum value.
        - tp (int): The time point corresponding to the maximum source estimate.
    """
    # Search for the index of the maximum value across the entire array
    _, index_tp = get_max_index(values_window)

    # Get the corresponding timepoint
    tp = window[0] + index_tp

    # Get the source estimate for this timepoint
    sestim_tp_values = values_window[:, index_tp]
    return sestim_tp_values, tp

def get_poa_window_sestim(values_window, window):
    """
    Computes the mean source estimate around the time point with the maximum value in the time course.
    
    Parameters:
    values_window (np.ndarray): A 2D numpy array where rows represent different sources and columns represent different time points.
    window (np.ndarray): A 1D numpy array representing the time window.
    
    Returns:
    tuple: A tuple containing:
        - sestim_window_values (np.ndarray): Mean source estimate values around the time point of maximum value.
        - tp (int): The time point corresponding to the maximum source estimate.
    """
    # Search for the index of the maximum value across the entire array
    _, index_tp = get_max_index(values_window)

    # Get the corresponding timepoint
    tp = window[0] + index_tp

    # Ensure the window around the maximum value is within the bounds of the array
    start = max(index_tp - 5, 0)
    end = min(index_tp + 5, values_window.shape[1])

    # Get the mean source estimate for this window
    sestim_window_values = np.mean(values_window[:, start:end], axis=1)
    return sestim_window_values, tp

def compute_window_indices(window, t_start, t_end, t_step, stc_length):
    """
    Compute the start and end indices for a time window, handling out-of-bounds cases.

    Parameters:
    window (tuple): The time window (start, end) in seconds.
    t_start (float): The start time of the data in seconds.
    t_end (float): The end time of the data in seconds.
    t_step (float): The time step between data points in seconds.
    stc_length (int): The number of time points in the data.

    Returns:
    tuple: The start and end indices for the window within the data.
    """
    # Calculate start index
    if window[0] >= t_start:
        start_idx = int((window[0] - t_start) / t_step)
    else:
        start_idx = 0
        warnings.warn(f"Window start {window[0]}s is before the first timepoint ({t_start}s). Capped to start of data.")

    # Calculate end index
    if window[1] <= t_end:
        end_idx = int((window[1] - t_start) / t_step) + 1
    else:
        end_idx = stc_length - 1
        warnings.warn(f"Window end {window[1]}s is beyond the last timepoint ({t_end}s). Capped to end of data.")

    return start_idx, end_idx