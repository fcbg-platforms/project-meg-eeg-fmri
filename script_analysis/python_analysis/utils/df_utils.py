"""
df_utils.py

This module provides utility functions for creating and managing DataFrames with hierarchical columns 
used in the analysis of neuroimaging and electrophysiological data. Specifically, the functions in this module 
are designed to create DataFrames for different types of data and analysis:

1. `subject_POA_df`: Creates an empty DataFrame with hierarchical columns for Peak of Activation (POA) data. 
   Includes columns for POA measurements (EEG, MEG, fMRI) and distance metrics for EEG and MEG.

2. `subject_COG_df`: Creates an empty DataFrame with hierarchical columns for Center of Gravity (COG) data.
   Includes columns for COG measurements (EEG, MEG, fMRI) and distance metrics for EEG and MEG.

3. `subject_overlap_df`: Creates an empty DataFrame with hierarchical columns for overlap data. 
   Includes columns for cluster size in mm^3 for different modalities (EEG, MEG, fMRI) and overlap metrics for EEG and MEG.

4. `all_subject_POA_df`: Creates an empty DataFrame with hierarchical columns for Peak of Activation (POA) data 
   across all subjects. Includes columns for subject information, POA measurements, and distance metrics.

5. `all_subject_COG_df`: Creates an empty DataFrame with hierarchical columns for Center of Gravity (COG) data 
   across all subjects. Includes columns for subject information, COG measurements, and distance metrics.

6. `all_subject_overlap_df`: Creates an empty DataFrame with hierarchical columns for overlap data across all subjects.
   Includes columns for subject information, cluster size in mm^3, and overlap metrics.

7. `reshape_dataframe(df)`: Reshapes and combines EEG and MEG data into a unified format with cleaned column names and modality indicators.

Each function returns a DataFrame with a MultiIndex structure, suitable for organizing and analyzing complex 
neuroimaging and electrophysiological datasets.
"""


import pandas as pd

def subject_POA_df():
    """
    Creates an empty DataFrame with hierarchical columns for Peak of Activation (POA) data.
    
    The DataFrame includes columns for general information and measurements related to:
    - POA (Peak of Activation) across different modalities (EEG, MEG, fMRI)
    - Distance metrics for EEG and MEG (mm)
    
    Returns:
        pd.DataFrame: An empty DataFrame with specified hierarchical columns for POA data.
    """
    # Define hierarchical columns
    columns = pd.MultiIndex.from_tuples([
        # General Information
        ('Info', 'task'),
        ('Info', 'condition'),
        ('Info', 'tpindex'),
        ('Info', 'fmridxEeg'),
        ('Info', 'fmridxMeg'),
        
        # POA (Peak of Activation) measurements        
        ('POA', 'eeg'),
        ('POA', 'fmrieeg'),
        ('POA', 'meg'),
        ('POA', 'fmrimeg'),
        
        # Distance Meg (Magnetoencephalography) metrics
        ('DistMeg', 'x'),
        ('DistMeg', 'y'),
        ('DistMeg', 'z'),
        ('DistMeg', 'norm'),
        ('DistEeg', 'x'),
        ('DistEeg', 'y'),
        ('DistEeg', 'z'),
        ('DistEeg', 'norm'),
  
    ])
    
    # Create an empty DataFrame with the specified hierarchical columns
    df = pd.DataFrame(columns=columns)
    
    return df

def subject_COG_df():
    """
    Creates an empty DataFrame with hierarchical columns for Center of Gravity (COG) data.
    
    The DataFrame includes columns for general information and measurements related to:
    - COG (Center of Gravity) across different modalities (EEG, MEG, fMRI)
    - Distance metrics for EEG and MEG (mm)
    
    Returns:
        pd.DataFrame: An empty DataFrame with specified hierarchical columns for COG data.
    """
    # Define hierarchical columns
    columns = pd.MultiIndex.from_tuples([
        # General Information
        ('Info', 'task'),
        ('Info', 'condition'),
        ('Info', 'tpindex'),
        ('Info', 'fmridxEeg'),
        ('Info', 'fmridxMeg'),
        
        # COG (Center of Gravity) measurements
        ('COG', 'eeg'),
        ('COG', 'fmrieeg'),
        ('COG', 'meg'),
        ('COG', 'fmrimeg'),
        

        # Distance Meg (Magnetoencephalography) metrics
        ('DistMeg', 'x'),
        ('DistMeg', 'y'),
        ('DistMeg', 'z'),
        ('DistMeg', 'norm'),
        ('DistEeg', 'x'),
        ('DistEeg', 'y'),
        ('DistEeg', 'z'),
        ('DistEeg', 'norm'),
  
    ])
    
    # Create an empty DataFrame with the specified hierarchical columns
    df = pd.DataFrame(columns=columns)
    
    return df

def subject_overlap_df():
    """
    Creates an empty DataFrame with hierarchical columns for overlap data.
    
    The DataFrame includes columns for general information and measurements related to:
    - Cluster size in mm^3 for different modalities (EEG, MEG, fMRI)
    - Overlap metrics for EEG and MEG (mm3)
    
    Returns:
        pd.DataFrame: An empty DataFrame with specified hierarchical columns for overlap data.
    """
    # Define hierarchical columns
    columns = pd.MultiIndex.from_tuples([
        # General Information
        ('Info', 'task'),
        ('Info', 'condition'),
        ('Info', 'tpindex'),
        ('Info', 'fmridxEeg'),
        ('Info', 'fmridxMeg'),
        
        # ClusterSize in mm3 measurements
        ('ClusterSizemm3', 'eeg'),
        ('ClusterSizemm3', 'fmrieeg'),
        ('ClusterSizemm3', 'meg'),
        ('ClusterSizemm3', 'fmrimeg'),

        # Overlap Eeg (Electroencephalography) metrics
        ('OverlapEeg', 'JaccardIdx'),
  
        # Overlap Meg (Magnetoencephalography) metrics
        ('OverlapMeg', 'JaccardIdx'),

    ])
    
    # Create an empty DataFrame with the specified hierarchical columns
    df = pd.DataFrame(columns=columns)
    
    return df

def all_subject_POA_df():
    """
    Creates an empty DataFrame with hierarchical columns for Peak of Activation (POA) data for all subjects.
    
    The DataFrame includes columns for general information about each subject and POA measurements across:
    - POA (Peak of Activation) modalities (EEG, MEG, fMRI)
    - Distance metrics for EEG and MEG (mm)
    
    Returns:
        pd.DataFrame: An empty DataFrame with specified hierarchical columns for POA data across all subjects.
    """
    # Define the columns
    columns = pd.MultiIndex.from_tuples([
        # General Information
        ('Info', 'SubjectName'),
        ('Info', 'task'),
        ('Info', 'condition'),
        ('Info', 'tpindex'),
        ('Info', 'fmridxEeg'),
        ('Info', 'fmridxMeg'),
        
        # POA (Peak of Activation) measurements        
        ('POA', 'eeg'),
        ('POA', 'fmrieeg'),
        ('POA', 'meg'),
        ('POA', 'fmrimeg'),
        
        # Distance Meg (Magnetoencephalography) metrics
        ('DistMeg', 'x'),
        ('DistMeg', 'y'),
        ('DistMeg', 'z'),
        ('DistMeg', 'norm'),
        ('DistEeg', 'x'),
        ('DistEeg', 'y'),
        ('DistEeg', 'z'),
        ('DistEeg', 'norm'),
  
    ])
    
    # Create an empty DataFrame with the specified hierarchical columns
    df = pd.DataFrame(columns=columns)
    
    return df

def all_subject_COG_df():
    """
    Creates an empty DataFrame with hierarchical columns for Center of Gravity (COG) data for all subjects.
    
    The DataFrame includes columns for general information about each subject and COG measurements across:
    - COG (Center of Gravity) modalities (EEG, MEG, fMRI)
    - Distance metrics for EEG and MEG (mm)
    
    Returns:
        pd.DataFrame: An empty DataFrame with specified hierarchical columns for COG data across all subjects.
    """
    # Define the columns
    columns = pd.MultiIndex.from_tuples([
        # General Information
        ('Info', 'SubjectName'),
        ('Info', 'task'),
        ('Info', 'condition'),
        ('Info', 'tpindex'),
        ('Info', 'fmridxEeg'),
        ('Info', 'fmridxMeg'),
        
        # COG (Center of Gravity) measurements
        ('COG', 'eeg'),
        ('COG', 'fmrieeg'),
        ('COG', 'meg'),
        ('COG', 'fmrimeg'),
        

        # Distance Meg (Magnetoencephalography) metrics
        ('DistMeg', 'x'),
        ('DistMeg', 'y'),
        ('DistMeg', 'z'),
        ('DistMeg', 'norm'),
        ('DistEeg', 'x'),
        ('DistEeg', 'y'),
        ('DistEeg', 'z'),
        ('DistEeg', 'norm'),
  
    ])

    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=columns)

    return df

def all_subject_overlap_df():
    """
    Creates an empty DataFrame with hierarchical columns for overlap data for all subjects.
    
    The DataFrame includes columns for general information about each subject and overlap measurements across:
    - Cluster size in mm^3 for different modalities (EEG, MEG, fMRI)
    - Overlap metrics for EEG and MEG (mm3)
     
    Returns:
        pd.DataFrame: An empty DataFrame with specified hierarchical columns for overlap data across all subjects.
    """
    # Define the columns
    columns = pd.MultiIndex.from_tuples([
        # General Information
        ('Info', 'SubjectName'),
        ('Info', 'task'),
        ('Info', 'condition'),
        ('Info', 'tpindex'),
        ('Info', 'fmridxEeg'),
        ('Info', 'fmridxMeg'),
        
        # ClusterSize in mm3 measurements
        ('ClusterSizemm3', 'eeg'),
        ('ClusterSizemm3', 'fmrieeg'),
        ('ClusterSizemm3', 'meg'),
        ('ClusterSizemm3', 'fmrimeg'),

        
        # Overlap Eeg (Electroencephalography) metrics
        ('OverlapEeg', 'JaccardIdx'),
  
        # Overlap Meg (Magnetoencephalography) metrics
        ('OverlapMeg', 'JaccardIdx'),

    ])

    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=columns)

    return df



def reshape_dataframe(df):
    """
    Reshape the given DataFrame to separate and combine EEG and MEG data,
    while cleaning up column names and adding a modality indicator.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing EEG and MEG data with various columns.

    Returns:
        pd.DataFrame: A DataFrame with EEG and MEG data combined, with cleaned column names and a modality indicator.
    """
    # Define a function to prepare DataFrames for EEG and MEG
    def prepare_modality_df(df, modality):
        """
        Prepare a DataFrame for a specific modality (EEG or MEG).
        
        Parameters:
            df (pd.DataFrame): The input DataFrame containing modality data.
            modality (str): The modality to filter for ('eeg' or 'meg').

        Returns:
            pd.DataFrame: A DataFrame filtered for the given modality with relevant columns and a modality indicator.
        """
        if modality == 'eeg':
            columns_to_keep = ['Info_SubjectName', 'Info_task', 'Info_condition', 'Info_tpindex', 
                               'Info_fmridxEeg', 'DistEeg_x', 'DistEeg_y', 'DistEeg_z', 'DistEeg_norm']
        elif modality == 'meg':
            columns_to_keep = ['Info_SubjectName', 'Info_task', 'Info_condition', 'Info_tpindex', 
                               'Info_fmridxMeg', 'DistMeg_x', 'DistMeg_y', 'DistMeg_z', 'DistMeg_norm']
        else:
            raise ValueError("Modality must be either 'eeg' or 'meg'.")

        # Check if all required columns are present
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        df_modality = df[columns_to_keep].copy()
        df_modality['modality'] = modality
        return df_modality

    # Define a function to clean column names
    def clean_column_names(df):
        """
        Clean the column names by removing modality-specific substrings.

        Parameters:
            df (pd.DataFrame): The DataFrame with modality-specific column names.

        Returns:
            pd.DataFrame: The DataFrame with cleaned column names.
        """
        df.columns = [col.replace('eeg', '').replace('meg', '').replace('Eeg', '').replace('Meg', '').strip('_') for col in df.columns]
        return df

    # Prepare DataFrames for EEG and MEG
    df_eeg = prepare_modality_df(df, 'eeg')
    df_meg = prepare_modality_df(df, 'meg')

    # Clean column names
    df_eeg_clean = clean_column_names(df_eeg)
    df_meg_clean = clean_column_names(df_meg)

    # Concatenate EEG and MEG DataFrames
    df_combined = pd.concat([df_eeg_clean, df_meg_clean], ignore_index=True)
    
    return df_combined
