"""dispersion_analysis.py

This module provides functions for performing dispersion analysis on EEG and MEG data. It includes methods for calculating and recording dispersion statistics using permutation-based methods and reshaping data for analysis. Additionally, it supports visualization of significant results.

Functions
-----------
compute_dispersion_analysis(df, analysis_type='POA', dist_metric='euclidean', nb_permutation=999, path_to_data=None)
    Performs a dispersion analysis on the provided data, computing distance dispersions for EEG and MEG modalities across conditions and time points. Results are stored in a DataFrame, and significant findings can be plotted.

significance_code(p_value)
    Returns a significance code based on the provided p-value for easy interpretation of statistical results.
"""

import pandas as pd
import numpy as np
from skbio.stats.distance import permdisp, DistanceMatrix
from scipy.spatial.distance import pdist, squareform
from utils.plot_utils import plot_pca_if_significant
from utils.df_utils import reshape_dataframe

def compute_dispersion_analysis(df, analysis_type = 'POA', dist_metric = 'euclidean', nb_permutation = 999, path_to_data = None):
    """
    Perform a dispersion analysis on the provided DataFrame, calculating and recording
    dispersion statistics based on different conditions and time points using `permdisp` from skbio.stats.distance. This function
    computes the distance dispersion for EEG and MEG data and stores the results in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data. Expected columns include:
            - 'Dist_x', 'Dist_y', 'Dist_z': Coordinates for distance calculations.
            - 'modality': Modality type (e.g., 'eeg' or 'meg').
            - 'Info_tpindex': Time point index.
            - 'Info_condition': Experimental condition.
        analysis_type (str): The type of data studied: POA or COG. Defaults to 'POA'
        dist_metric (str, optional): The distance metric to use for dispersion analysis. Defaults to 'euclidean'.
        nb_permutation (int, optional): The number of permutations to be used in the dispersion test. Defaults to 999.
        path_to_data (str, optional): Directory path to save plots. If None, plots will not be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the dispersion analysis. Columns include:
            - 'condition': Experimental condition.
            - 'tp': Time point index.
            - 'method_name': Name of the method used for dispersion calculation.
            - 'test_statistic_name': Name of the test statistic.
            - 'sample_size': Number of samples.
            - 'number_of_groups': Number of groups compared.
            - 'test_statistic': Value of the test statistic.
            - 'p_value': p-value of the test.
            - 'number_of_permutations': Number of permutations used in the analysis.
    """
    
    results_dispersion_analysis = pd.DataFrame(columns=[
        'condition', 'tp', 'method_name', 'test_statistic_name', 
        'sample_size', 'number_of_groups', 'test_statistic', 
        'p_value', 'number_of_permutations'
    ])

    def add_results(condition, tp, permdisp_results):
        """
        Add the results of the dispersion analysis to the results DataFrame.

        Parameters:
            condition (str): The experimental condition (e.g., 'face', 'scene').
            tp (int or str): The time point index or 'all' if applicable.
            permdisp_results (dict): Results from the permutation dispersion analysis. Expected keys:
                - 'method name': Name of the method used.
                - 'test statistic name': Name of the test statistic.
                - 'sample size': Number of samples used.
                - 'number of groups': Number of groups compared.
                - 'test statistic': Value of the test statistic.
                - 'p-value': p-value of the test.
                - 'number of permutations': Number of permutations performed.

        NB:
        Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
        """
        nonlocal results_dispersion_analysis
        # Create a new row
        new_row = pd.DataFrame({
            'condition': condition,
            'tp': tp,
            'method_name': permdisp_results['method name'],
            'test_statistic_name': permdisp_results['test statistic name'],
            'sample_size': permdisp_results['sample size'],
            'number_of_groups': permdisp_results['number of groups'],
            'test_statistic': permdisp_results['test statistic'],
            'p_value': [f"{permdisp_results['p-value']:.3f} ({significance_code(permdisp_results['p-value'])})"],
            'number_of_permutations': permdisp_results['number of permutations']}, index = [len(results_dispersion_analysis)])
    
        # Handle the case where both DataFrames might be empty
        if results_dispersion_analysis.empty:
            results_dispersion_analysis = new_row
        else:
            # Concatenate while ensuring correct data types and avoiding all-NA issues
            results_dispersion_analysis = pd.concat([results_dispersion_analysis, new_row], ignore_index=True)

    # Compute dispersion for the entire dataset
    all_poa_df = df[['Dist_x', 'Dist_y', 'Dist_z']]
    all_poa_df = all_poa_df.dropna()
    modalities = df[['modality']]
    dist_matrix = DistanceMatrix(squareform(pdist(all_poa_df, metric=dist_metric)), ids=modalities.index)
    stats_results = permdisp(dist_matrix, modalities['modality'], permutations = nb_permutation)
    add_results('all', 'all', stats_results)
    plot_pca_if_significant(modalities, all_poa_df, stats_results['p-value'], 'all', 'all', alpha=0.05, path_to_data = path_to_data, analysis_type = analysis_type)

    # Compute dispersion for each time point
    for tp in range(3):
        print(f'  Analyzing tp {tp}.')
        filtered_df = df[df['Info_tpindex'] == tp]
        all_poa_df = filtered_df[['Dist_x', 'Dist_y', 'Dist_z']]
        all_poa_df = all_poa_df.dropna()
        modalities = filtered_df[['modality']]
        dist_matrix = DistanceMatrix(squareform(pdist(all_poa_df, metric=dist_metric)), ids=modalities.index)
        stats_results = permdisp(dist_matrix, modalities['modality'], permutations = nb_permutation)
        add_results('all', tp, stats_results)
        plot_pca_if_significant(modalities, all_poa_df, stats_results['p-value'], tp, 'all', alpha=0.05, path_to_data = path_to_data, analysis_type = analysis_type)

    # Compute dispersion for each condition
    all_conditions = np.unique(df['Info_condition'])
    for condi in all_conditions:
        print(f'  Analyzing condition {condi}.')
        filtered_df = df[df['Info_condition'] == condi]
        all_poa_df = filtered_df[['Dist_x', 'Dist_y', 'Dist_z']]
        all_poa_df = all_poa_df.dropna()
        modalities = filtered_df[['modality']]
        dist_matrix = DistanceMatrix(squareform(pdist(all_poa_df, metric=dist_metric)), ids=modalities.index)
        stats_results = permdisp(dist_matrix, modalities['modality'], permutations = nb_permutation)
        add_results(condi, 'all', stats_results)
        plot_pca_if_significant(modalities, all_poa_df, stats_results['p-value'], 'all', condi, alpha=0.05, path_to_data = path_to_data, analysis_type = analysis_type)


        # Compute dispersion for each time point within each condition
        for tp in range(3):
            print(f'    Analyzing tp {tp}.')
            filtered_df_tp = filtered_df[filtered_df['Info_tpindex'] == tp]
            all_poa_df_tp = filtered_df_tp[['Dist_x', 'Dist_y', 'Dist_z']]
            all_poa_df_tp = all_poa_df_tp.dropna()
            
            modalities_tp = filtered_df_tp[['modality']]
            dist_matrix_tp = DistanceMatrix(squareform(pdist(all_poa_df_tp, metric=dist_metric)), ids=modalities_tp.index)
            stats_results_tp = permdisp(dist_matrix_tp, modalities_tp['modality'], permutations = nb_permutation)
            add_results(condi, tp, stats_results_tp)
            plot_pca_if_significant(modalities_tp, all_poa_df_tp, stats_results_tp['p-value'], tp, condi, alpha=0.05, path_to_data = path_to_data, analysis_type = analysis_type)

    return results_dispersion_analysis

def significance_code(p_value):
    """
    Return significance code based on the p-value.
    
    Parameters:
        p_value (float): The p-value to be converted.
        
    Returns:
        str: Significance code.
    """
    if p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    elif p_value <= 0.1:
        return '.'
    else:
        return ' '