"""plot_utils.py

This module provides functions for performing Principal Component Analysis (PCA) on neuroimaging data, specifically for evaluating modality differences and plotting significant results. Additionally, it includes utilities for plotting source time courses (STC) and computing spatial medians of PCA components.

Functions
-----------
spatial_median(points)
    Computes the spatial median (geometric median) of a set of points.

calculate_spatial_median(group)
    Calculates the spatial median for a group of PCA results for a specific modality.

plot_pca_if_significant(modalities, all_poa_df, p_value, tp, condi, alpha=0.05, path_to_data=None, analysis_type='POA')
    Performs PCA and plots the first two principal components if the PERMDISP p-value is significant. Saves the plot if an output directory is specified.

plot_stc(subject, directory, subjects_dir, pos)
    Plots source time course (STC) data for a given subject, task, and event. Generates both static and interactive 3D plots for visualization.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from pathlib import Path

from mne import read_forward_solution, read_source_estimate, convert_forward_solution, Info, SourceSpaces

def spatial_median(points):
    """
    Compute the spatial median (geometric median) of a set of points.
    
    Parameters:
        points (np.ndarray): Array of points with shape (n_samples, n_features).
        
    Returns:
        np.ndarray: The spatial median of the input points.
    """
    median = np.mean(points, axis=0)
    
    for _ in range(100):
        distances = np.linalg.norm(points - median, axis=1)
        nonzero_distances = distances > 0
        weights = np.where(nonzero_distances, 1 / distances, 0)
        weighted_sum = np.dot(weights, points) / np.sum(weights)
        
        if np.all(median == weighted_sum):
            break
        median = weighted_sum
    
    return median

def calculate_spatial_median(group):
    """
    Calculate the spatial median for a group of PCA results.
    
    Parameters:
        group (pd.DataFrame): DataFrame containing PCA results for a single modality.
        
    Returns:
        pd.Series: The spatial median for the given group.
    """
    median = spatial_median(group[['PC1', 'PC2']].values)
    return pd.Series(median, index=['PC1_median', 'PC2_median'])


def plot_pca_if_significant(modalities, all_poa_df, p_value, tp, condi, alpha=0.05, path_to_data = None, analysis_type = 'POA'):
    """
    Perform PCA and plot the first two principal coordinates if the PERMDISP p-value is significant.
    
    Parameters:
        modalities (pd.DataFrame): DataFrame containing modality information.
        all_poa_df (pd.DataFrame): DataFrame containing the features for PCA.
        p_value (float): The p-value from the PERMDISP test.
        tp (str): Identifier for the time point.
        condi (str): Identifier for the condition.
        alpha (float): Significance level for determining if the p-value is significant (default is 0.05)
        path_to_data (str, optional): Directory path to save the plot. If None, the plot will not be saved.
        analysis_type (str): The type of data studied: POA or COG. Defaults to 'POA'
    """
    if p_value >= alpha:
        return
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_poa_df[['Dist_x', 'Dist_y', 'Dist_z']])
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
    pca_df['modality'] = modalities['modality'].values
    
    # Loadings for each principal component
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=['Dist_x', 'Dist_y', 'Dist_z'])
    most_influential = loadings.idxmax()
    
    # Calculate spatial medians for each modality
    medians = pca_df.groupby('modality').apply(calculate_spatial_median).reset_index()
    medians.columns = ['modality', 'PC1_median', 'PC2_median']

    # Define color palette
    color_palette = sns.color_palette('Set1', n_colors=len(pca_df['modality'].unique()))
    color_mapping = dict(zip(pca_df['modality'].unique(), color_palette))
    
    # Plot the PCA results
    plt.figure(figsize=(12, 8))
    
    # Plot the points
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='modality', palette=color_mapping, edgecolor='w', s=100, alpha=0.7)
    
    # Plot spatial medians and lines to the spatial medians
    for _, row in medians.iterrows():
        modality = row['modality']
        plt.scatter(row['PC1_median'], row['PC2_median'], color='w', s=200, marker='D', edgecolor=color_mapping[modality], linewidth=2, label=f'{modality} Median')
        
        subset = pca_df[pca_df['modality'] == modality]
        if not subset.empty:
            # Plot lines from each point to its spatial median
            for _, point in subset.iterrows():
                plt.plot([point['PC1'], row['PC1_median']], [point['PC2'], row['PC2_median']], color=color_mapping[modality], linestyle='-', linewidth=1, alpha=0.6)
            
            # Compute and plot the contour around the furthest points
            distances = np.linalg.norm(subset[['PC1', 'PC2']].values - np.array([row['PC1_median'], row['PC2_median']]), axis=1)
            furthest_points = subset.iloc[np.argsort(distances)[-20:]]  # Consider top 20 furthest points
            if len(furthest_points) >= 3:
                hull = ConvexHull(furthest_points[['PC1', 'PC2']].values)
                for simplex in hull.simplices:
                    plt.plot(furthest_points.iloc[simplex, 0], furthest_points.iloc[simplex, 1], color=color_mapping[modality], linestyle='-', linewidth=2, alpha=0.5)
    
    # Set titles and labels with increased font sizes
    plt.title(f'{analysis_type} Dispersion Test: tp = {tp}, condi = {condi}, p_value = {p_value}', fontsize=16)
    plt.xlabel(f'PC1: {most_influential["PC1"]}', fontsize=15)
    plt.ylabel(f'PC2: {most_influential["PC2"]}', fontsize=15)
    plt.legend(title='Modality', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(False)
    plt.tight_layout()  # Adjust layout to make room for legend
    
    # Save the plot as an image file if path_to_data is provided
    if path_to_data is not None:
        path_save = Path(path_to_data) / 'pca_plot_python'
        path_save.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_save / f'all_subjects_analysis-{analysis_type}_modality_comparison_analysis-dispersion_pca_plot_tp-{tp}_condi-{condi}_python.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()


def plot_stc(subject, directory, subjects_dir, pos):
    """
    Plot source time course (STC) data for a given subject, task, and event.
    
    Parameters:
    -----------
    subject : str
        The subject identifier (e.g., '01').
    directory : pathlib.Path
        The base directory containing the STC files, forward solutions, and other data.
    subjects_dir : str
        The FreeSurfer subjects directory path.
    pos : str
        The position identifier, used to filter the appropriate files.

    Notes:
    ------
    - The function will search for STC files in the "Sources/stc" directory, filtering by the specified 
      `pos`.
    - For each matching STC file, it reads the corresponding forward solution, converts it, and then 
      plots the source estimate.
    - Two types of plots are generated for each STC: a static plot and a 3D interactive plot.
    - The function pauses after each plot, allowing the user to view it before moving to the next one.
    """
    stc_dir = directory / "Sources" / "stc"
    
    # Iterate over STC files in the directory
    for stc_file in stc_dir.iterdir():
        # Filter for the correct file type and position
        if stc_file.suffix != ".h5" or f"pos_{pos}" not in stc_file.name:
            continue
        
        # Extract task and event information from the filename
        try:
            task = stc_file.stem.split("task_")[1].split("_")[0]
            event = stc_file.stem.split("-epo-")[1].split("-pos")[0]
        except IndexError:
            print(f"Filename {stc_file.name} does not follow the expected format. Skipping.")
            continue
        
        print(f"Task: {task}\tEvent: {event}")
        
        # Read forward solution file
        fwd_file = directory / "Sources" / "forward" / f"sub_{subject}-pos_{pos}-fwd.h5"
        fwd = read_forward_solution(fwd_file)
        
        # Ensure forward solution components are correctly formatted
        fwd["info"] = Info(fwd["info"])
        fwd["src"] = SourceSpaces(fwd["src"])
        
        # Convert forward solution to desired orientation and settings
        fwd = convert_forward_solution(fwd, force_fixed=False, surf_ori=True, copy=False)
        
        # Read source estimate
        stc = read_source_estimate(stc_file)
        
        # Plot the static source estimate
        stc.plot(fwd["src"], subject=subject, subjects_dir=subjects_dir, show=False)
        
        # Plot the 3D interactive source estimate
        stc.plot_3d(subject=subject, subjects_dir=subjects_dir, src=fwd["src"])
        
        # Show the plot and wait for user interaction
        plt.show()
        input("Press Enter to continue to the next file...")