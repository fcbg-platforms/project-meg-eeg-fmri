"""
nifti_utils.py

This module provides utility functions for working with NIfTI file headers, specifically for extracting degrees of freedom (DOF) and threshold values from the header descriptions. It includes functions to decode and parse header fields to retrieve specific numeric values.

Functions:
- extract_dof_from_niftiheader(niftiheader)
    Extract the degree of freedom (DOF) from the 'descrip' field of a NIfTI file header.

- extract_threshold(zscore_thres)
    Extract the threshold value from the 'descrip' field of a NIfTI file header.
"""

import re
import nibabel as nib

def extract_dof_from_niftiheader(niftiheader):
    """
    Extracts the degree of freedom (DOF) from the 'descrip' field of a NIfTI file header.

    Parameters:
        niftiheader (nibabel.nifti1.Nifti1Header): The NIfTI file header.

    Returns:
        float: The extracted degree of freedom (DOF) value.

    Raises:
        ValueError: If the DOF pattern is not found in the 'descrip' field.
    """
    # Access the descrip field
    descrip = niftiheader['descrip']

    # Decode the bytes to a string (if necessary)
    descrip_str = descrip.tobytes().decode('utf-8').strip()

    # Use a regular expression to extract the number
    match = re.search(r'\[(\d+\.\d+)\]', descrip_str)
    if match:
        value = float(match.group(1))
        return value
    else:
        raise ValueError('No Degree of Freedom found in the descrip field.')


def extract_threshold(zscore_thres):
    """
    Extract the threshold value from the 'descrip' field of a NIfTI file header.

    Parameters
    ----------
    zscore_thres : nib.Nifti1Image
        The NIfTI image object from which to extract the threshold value.

    Returns
    -------
    float
        The threshold value extracted from the NIfTI header description.

    Raises
    ------
    ValueError
        If no threshold value is found in the NIfTI header description.
    """
    ZscoreHeaderDescrip = zscore_thres.header['descrip']
    ZscoreHeaderDescrip_str = ZscoreHeaderDescrip.tobytes().decode('utf-8').strip()
    match = re.search(r"threshold = ([0-9.]+)", ZscoreHeaderDescrip_str)
    if match:
        return float(match.group(1))
    else:
        raise ValueError('No threshold value found in the NIfTI header description.')