"""
generation_compa_df_utils.py
This script provides functions for analyzing and comparing neuroimaging data from fMRI, EEG, and MEG modalities.

Key Functions:
1. `comparison_EEG_MEG_fMRI_df_fill`: Main function that compares EEG, MEG, and fMRI data for specific tasks and conditions. It extracts POA (Point of Activation) coordinates, calculates weighted Center of Gravity (COG) coordinates, and computes distances and overlaps between MRI and EEG (and MEG) data. Results are saved into Excel files.
2. `extract_cluster_info`: Extracts cluster information including coordinates, sizes, and cluster maps from neuroimaging data files.
3. `found_wcog`: Computes the Weighted Center of Gravity (WCOG) for specified clusters in a 3D voxel map using modality data for weighting.
4. `compute_overlap`: Calculates the overlap between two cluster maps using the Jaccard index.
5. `calculate_distances`: Computes Euclidean and axis-specific distances between 3D points.
6. `found_cog`: Calculates the Center of Gravity (COG) for specified clusters in a 3D voxel map.
7. `fill_sub_df`: Updates DataFrames with analysis results for individual subjects, including POA coordinates, COG coordinates, distances, and overlap metrics.
"""


import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn.reporting import get_clusters_table

from utils.nifti_utils import extract_threshold
from utils.utils import save_df_to_csv, found_condition_file
from utils.mask_utils import masking_data_w_roi

# Main Function
def comparison_EEG_MEG_fMRI_df_fill(sub_derivatives_outdir, sub, task, task_conditions, df_all_sub_, df_sub_, corr_param, thres_param, eeg_stc_algo, pos, logger, fmri_comparison_clusters = [1, 2, 3], do_brain_masking = False, roi_masking = False, atlas = None):
    """
    Compare EEG, MEG, and fMRI data for specified tasks and conditions.

    This function performs the following steps:
    1. Defines paths for fMRI, EEG, and MEG results based on the provided subject and task.
    2. Loops through each condition and for the given task to perform the following analyses:
        - Extracts the POA (Point of Activation) coordinates and cluster information from MRI results.
        - Computes the wCOG (weighted Center of Gravity) coordinates for the top clusters in MRI.
        - For each EEG and MEG files, extracts POA coordinates and cluster information and computes distances to the MRI POA coordinates.
        - Calculates distances between MRI and EEG (resp.MEG) wCOG coordinates.
        - Computes overlap metrics between MRI and EEG (resp.MEG) clusters.
    3. Populates and updates dataframes with analysis results for each task and condition.
    4. Saves the subject-specific and combined dataframes to Excel files for further analysis.

    Parameters:
    - sub_derivatives_outdir (str): derivatives directory path where data files are stored for the running subject.
    - sub (str): Subject identifier.
    - task_conditions (list): List of the conditions of a given task.
    - task (str): The running task
    - df_all_sub_ (dict): Dictionary of dataframes storing combined results for all subjects, keyed by analysis type.
    - df_sub_ (dict): Dictionary of dataframes storing subject-specific results, keyed by analysis type.
    - corr_param (object): Object containing parameters for fMRI correction (e.g., correction method, alpha, cluster size, etc.).
    - thres_param (dict): Dictionary containing threshold parameters for EEG and MEG data.
    - eeg_stc_algo (str): The algorithm used to compute eeg source estimates.
    - pos (int): the MEG stc resolution
    - logger (logging.Logger): Logger object for recording errors and information.
    - fmri_comparison_clusters (list): the fmri cluster to which the eeg and meg first cluster will be compared to
    - do_brain_masking (bool, optional): If True, a brain mask is applied to exclude non-brain regions from the NIFTI interpolated images. Default is False.
    - roi_masking (bool, optional): If True, the map are masked by the roi associated to the task, for the atlas `atlas`. Default is False.
    - atlas (str, optional): The name of the atlas to use when roi_masking. Default is None
    """
    
    corr_method = corr_param.fMRI_CORRECTION_METHOD
    mri_alpha = corr_param.fMRI_ALPHA
    mri_cluster_thres = corr_param.fMRI_CLUSTER_THRES
    mri_twosided = corr_param.fMRI_TWOSIDED

    # Def path to the mri results
    mri_path = sub_derivatives_outdir / 'func' / f'task-{task}' / f'corr-{corr_method}'

    # # Def path to the eeg results
    eeg_path = sub_derivatives_outdir/ 'eeg' / 'stc_interpolated' / f'task-{task}'
    
    # Def path to the meg results
    meg_path = sub_derivatives_outdir / 'meg' / 'stc_interpolated' / f'task-{task}'

    # Loop through conditions
    for condi in task_conditions:
        logger.info(f"    Analyzing condition {condi}.")
        print(f"    Analyzing condition {condi}.")

        # Define file patterns
        eeg_pattern = eeg_path / f"{sub}_task-{task}_condition-{condi}_desc-eeg-stcinterpol_tp-*_stat-{eeg_stc_algo}_statmap_resamp_masked_topercent-{thres_param['EEG']['alpha']}_cluster-{thres_param['EEG']['cluster_thresh']}.nii.gz"
        
        if do_brain_masking == False:
            meg_pattern = meg_path / f"{sub}_task-{task}_condition-{condi}_desc-meg-stcinterpol_tp-*_pos-{pos}_stat-sLORETA_statmap_resamp_topercent-{thres_param['MEG']['alpha']}_cluster-{thres_param['MEG']['cluster_thresh']}.nii.gz"
        else:
            meg_path / f"{sub}_task-{task}_condition-{condi}_desc-meg-stcinterpol_tp-*_pos-{pos}_stat-sLORETA_statmap_resampmasked_topercent-{thres_param['MEG']['alpha']}_cluster-{thres_param['MEG']['cluster_thresh']}.nii.gz"
        
        mri_file = mri_path / f"{sub}_task-{task}_contrast-{condi}_desc-stat-z_statmap_masked_corr-{corr_method}_alpha-{mri_alpha}_cluster-{mri_cluster_thres}_twosided-{mri_twosided}.nii.gz"
                    
        # Find EEG and MEG files
        try:
            eeg_files = found_condition_file(eeg_pattern)
            meg_files = found_condition_file(meg_pattern)
        except Exception as e:
            # Handle any exceptions during files searching
            logger.error(f"      {e}")
            raise(e)

        if roi_masking:
            try:
                mri_file = masking_data_w_roi(mri_file, sub_derivatives_outdir, sub, task, condi, atlas)
                logger.info(f"      Masking by the {task} roi of {mri_file} done ({atlas}).")
            except Exception as e:
                logger.error(f"      {e}.")
                raise(e)

        # Get the POA coordinates and the cluster_maps of the MRI contrast
        try:
            mri_poa_coords, mri_clusters_size, mri_cluster_map = extract_cluster_info(mri_file, mri_cluster_thres, cluster_ids = fmri_comparison_clusters, n_clusters=len(fmri_comparison_clusters))
            logger.info(f"      MRI POA coordinates computed.")
        except Exception as e:
            # Handle any exceptions during fMRI clusters study
            logger.error(f"      {e}")
            raise(e)

        # Compute the COG for the chosen MRI clusters
        try:
            mri_cogs_coords = found_wcog(mri_cluster_map[0], mri_file, cog_index = fmri_comparison_clusters)
            logger.info(f"      MRI COG coordinates computed.")
        except Exception as e:
            # Handle any exceptions during COG computation
            logger.error(f"      {e}")
            mri_cogs_coords = np.array([[np.nan, np.nan, np.nan]]) # Handle as needed
        
        # Loop through EEG amd MEG files
        for tp, (eeg_file, meg_file) in enumerate(zip(eeg_files, meg_files)):
            logger.info(f"      Analyzing tp {tp}:")

            if roi_masking:
                try:
                    eeg_file = masking_data_w_roi(eeg_file, sub_derivatives_outdir, sub, task, condi, atlas)
                    logger.info(f"      Masking by the {task} roi of {eeg_file} done ({atlas}).")
                except Exception as e:
                    logger.error(f"      {e}.")
                    raise(e)
                try:
                    meg_file = masking_data_w_roi(meg_file, sub_derivatives_outdir, sub, task, condi, atlas)
                    logger.info(f"      Masking by the {task} roi of {meg_file} done ({atlas}).")
                except Exception as e:
                    logger.error(f"      {e}.")
                    raise(e)

            # Get the POA coordinates
            eeg_poa_coords, eeg_clusters_size, eeg_cluster_map = extract_cluster_info(eeg_file, thres_param['EEG']['cluster_thresh'], cluster_ids=[1], n_clusters=1)
            logger.info(f"        EEG POA coordinates computed.")
            meg_poa_coords, meg_clusters_size, meg_cluster_map = extract_cluster_info(meg_file, thres_param['MEG']['cluster_thresh'], cluster_ids=[1], n_clusters=1)
            logger.info(f"        MEG POA coordinates computed.")
            
            # Compute the POA distances per coordinates
            eeg_POA_dist = calculate_distances(mri_poa_coords, eeg_poa_coords, fmri_comparison_clusters)
            logger.info(f"        EEG-fMRI POA distances computed.")
            meg_POA_dist = calculate_distances(mri_poa_coords, meg_poa_coords, fmri_comparison_clusters)
            logger.info(f"        MEG-fMRI POA distances computed.")
            
            # Compute the COG of the different clusters
            try:
                eeg_cog_coords = found_wcog(eeg_cluster_map[0], eeg_file, cog_index = [1])
                logger.info(f"        EEG COG coordinates computed.")
            except Exception as e:
                # Handle any exceptions that occur during overlap computation
                logger.error(f"        {e}.")
                eeg_cog_coords = np.array([[np.nan, np.nan, np.nan]])
            try:
                meg_cog_coords = found_wcog(meg_cluster_map[0], meg_file, cog_index = [1])
                logger.info(f"        MEG COG coordinates computed.")
            except Exception as e:
                # Handle any exceptions that occur during overlap computation
                logger.error(f"        {e}.")
                meg_cog_coords = np.array([[np.nan, np.nan, np.nan]])

            # Compute the COG distances
            eeg_COG_dist = calculate_distances(mri_cogs_coords, eeg_cog_coords, fmri_comparison_clusters)
            logger.info(f"        EEG-fMRI COG distances computed.")
            meg_COG_dist = calculate_distances(mri_cogs_coords, meg_cog_coords, fmri_comparison_clusters)
            logger.info(f"        MEG-fMRI COG distances computed.")
    

            # Compute the overlap size between modalities
            try:
                # Compute the overlap between the two closest clusters
                eeg_jaccardidx = compute_overlap(mri_cluster_map[0], eeg_cluster_map[0], cluster_value = eeg_COG_dist['min_index'])
                logger.info(f"        EEG-fMRI overlap computed.")

            except Exception as e:
                # Handle any exceptions that occur during overlap computation
                logger.error(f"        {e}")
                eeg_jaccardidx = {'jaccard_index': np.nan, 'cluster_index': np.nan}

            try:
                # Compute the overlap between the two closest clusters
                meg_jaccardidx = compute_overlap(mri_cluster_map[0], meg_cluster_map[0], cluster_value = meg_COG_dist['min_index'])
                logger.info(f"        MEG-fMRI overlap computed.")
            except Exception as e:
                # Handle any exceptions that occur during overlap computation
                logger.error(f"        {e}")
                meg_jaccardidx = {'jaccard_index': np.nan, 'cluster_index': np.nan}
            
           
            # Define the types of analysis and corresponding data
            analysis_types = ['POA', 'COG', 'Overlap']
            modal_data_list = [
                (mri_poa_coords, eeg_poa_coords, meg_poa_coords),
                (mri_cogs_coords, eeg_cog_coords, meg_cog_coords),
                (mri_clusters_size, eeg_clusters_size, meg_clusters_size)
            ]
            comparison_data_list = [
                (eeg_POA_dist, meg_POA_dist),
                (eeg_COG_dist, meg_COG_dist),
                (eeg_jaccardidx, meg_jaccardidx)
            ]

            # Update dataframes with analysis results
            for analysis_type, modal_data, comparison_data in zip(analysis_types, modal_data_list, comparison_data_list):   
                df_sub_[analysis_type] = fill_sub_df(df_sub_[analysis_type], task, condi, tp, modal_data, comparison_data, analysis_type, fmri_comparison_clusters)
                df_all_sub_[analysis_type] = fill_all_sub_df(df_all_sub_[analysis_type], sub, task, condi, tp, modal_data, comparison_data, analysis_type, fmri_comparison_clusters)

    # Save the dataframes after each task
    sub_outdir = sub_derivatives_outdir / 'results'
    general_outdir = sub_derivatives_outdir.parent / 'results'
    
    for analysis_type in analysis_types:
        # Save subject-specific DataFrame
        sub_outdir_type = sub_outdir / analysis_type 
        sub_outdir_type.mkdir(parents=True, exist_ok=True)
        save_df_to_csv(df_sub_[analysis_type], sub_outdir_type / f"{sub}_analysis-{analysis_type}_modality_comparison.csv")
        logger.info(f"      {analysis_type} {sub} dataframe saved at {sub_df_path}")
    
        # Save combined DataFrame for all subjects
        general_outdir_type = general_outdir / analysis_type
        general_outdir_type.mkdir(parents=True, exist_ok=True)
        save_df_to_csv(df_all_sub_[analysis_type], general_outdir_type / f"all_subjects_analysis-{analysis_type}_modality_comparison.csv")
        logger.info(f"      {analysis_type} All Subjects dataframe saved at {sub_df_path}")



def extract_cluster_info(file_path, cluster_threshold, cluster_ids=[1, 2, 3], n_clusters=3):
    """
    Extracts cluster information from a given file.

    Parameters:
        file_path (str, Path): Path to the data file (MRI, EEG, or MEG).
        cluster_threshold (float): Threshold for cluster extraction.
        cluster_ids (list): List of cluster IDs to filter.
        n_clusters (int): Number of clusters to extract coordinates for.

    Returns:
        - Cluster coordinates for the specified number of clusters
        - Cluster sizes for specified cluster IDs
        - Cluster map (label map)
    """
    # Load data 
    data = nib.load(file_path)

    # Extract the statistical threshold
    data_thres = extract_threshold(data)

    # Get clusters table and cluster map
    table, cluster_map = get_clusters_table(
        data,
        stat_threshold=data_thres,
        cluster_threshold=cluster_threshold,
        return_label_maps=True
    )

    # For the first few clusters, get coordinates
    num_clusters = min(len(table), n_clusters)
    coords = table.loc[range(num_clusters), ["X", "Y", "Z"]].values
    coords = np.array([[np.nan, np.nan, np.nan]]) if len(coords) == 0 else coords

    # Filter DataFrame for specific cluster IDs
    filtered_table = table[table['Cluster ID'].isin(cluster_ids)]

    # Extract the 'Cluster Size (mm3)' column
    cluster_sizes = filtered_table.loc[:, 'Cluster Size (mm3)'].values

    return coords, cluster_sizes, cluster_map

def found_wcog(cluster_map, modality_file, cog_index=[1]):
    """
    Calculate the Weighted Center of Gravity (WCOG) for specified clusters in a 3D voxel map
    using a modality file for weighting.

    Parameters:
    - cluster_map: A NIfTI image object or similar with a method `get_fdata()` 
      that returns a 3D numpy array where clusters are labeled with integers.
    - modality_file: A file path to a NIfTI image containing the modality data
      to be used for weighting the clusters.
    - cog_index: A list of integers where each integer corresponds to a specific
      cluster label in the cluster_map. Default is [1].

    Returns:
    - A numpy array of shape (n_clusters, 3) where each row corresponds to the WCOG
      of a cluster, with columns for the x, y, and z coordinates.
    """
    
    # Load cluster map and modality data
    cluster_data = cluster_map.get_fdata()
    modality_data = nib.load(modality_file).get_fdata()

    # Initialize list to store WCOG coordinates for each cluster
    WCOG = []

    for cluster_label in cog_index:
        # Get the voxel coordinates for the current cluster
        cluster_voxels = np.where(cluster_data == cluster_label)
        
        # Check if the cluster has any voxels
        if not cluster_voxels[0].size:
            WCOG.append([np.nan, np.nan, np.nan])
            continue
        
        # Extract modality data for the current cluster
        weights = modality_data[cluster_voxels]
        # Ensure weights are a 1D array
        weights = np.ravel(weights)
        
        # Calculate total weight
        total_weight = np.sum(weights)
        
        if total_weight == 0:
            WCOG.append([np.nan, np.nan, np.nan])
            continue

        # Calculate weighted center of gravity for each dimension      
        COGx = np.sum(np.array(cluster_voxels[0]) * weights) / total_weight
        COGy = np.sum(np.array(cluster_voxels[1]) * weights) / total_weight
        COGz = np.sum(np.array(cluster_voxels[2]) * weights) / total_weight

        # Pass in world coordinate
        [COGx, COGy, COGz] = nib.affines.apply_affine(cluster_map.affine, [COGx, COGy, COGz])

        # Append the WCOG coordinates to the list
        WCOG.append([COGx, COGy, COGz])

    # Convert the list to a numpy array and return
    return np.array(WCOG)

def compute_overlap(cluster_map1, cluster_map2, cluster_value = 1):
    """
    Compute the overlap size between two cluster maps for specified cluster indices.

    Parameters:
    - cluster_map1: A NIfTI image object or similar with a method `get_fdata()` 
      that returns a 3D numpy array for the first cluster map.
    - cluster_map2: A NIfTI image object or similar with a method `get_fdata()` 
      that returns a 3D numpy array for the second cluster map.
    - cluster_value: the cluster value of interest (the closest one). Default is 1.

    Returns:
    -  A Jaccard index, the number of overlapping voxels divided by the union of the sizes of the
      clusters in both maps.
    """
    
    # Load cluster data from both maps
    cluster_data1 = cluster_map1.get_fdata()
    cluster_data2 = cluster_map2.get_fdata()

    # Check if both maps have the same resolution
    voxel_size1 = np.prod(cluster_map1.header.get_zooms())
    voxel_size2 = np.prod(cluster_map2.header.get_zooms())
    
    if voxel_size1 != voxel_size2:
        raise ValueError("Both maps must have the same resolution.")

    # Get coordinates where both clusters overlap
    overlap_voxels = (cluster_data1 == cluster_value) & (cluster_data2 == 1)
    
    # Calculate the number of overlapping voxels
    overlap_size = np.sum(overlap_voxels) * voxel_size1

    # Calculate the size of the clusters
    cluster1_size = np.sum(cluster_data1 == cluster_value) * voxel_size1
    cluster2_size = np.sum(cluster_data2 == 1) * voxel_size2

    # Calculate the union size
    union_size = cluster1_size + cluster2_size - overlap_size
    
    if union_size == 0:
        jaccard_index = 0
    else:
        jaccard_index = overlap_size / union_size
    
    jaccard_index = {'jaccard_index': round(jaccard_index, 2), 'cluster_index': cluster_value}

    return jaccard_index
    
def calculate_distances(mri_points, eg_point, fmri_comparison_clusters):
    """
    Calculate the Euclidean distance and distances in each dimension between two 3D points.

    Parameters:
        mri_poins (tuple or list): Coordinates of the MRI points (x1, y1, z1).
        eg_point (tuple or list): Coordinates of the EEG or MEG point (x2, y2, z2).
        fmri_comparison_clusters (list): Indices of clusters that originate from the mri_points.

    Returns:
        dict with in it: 
            - Euclidean distance between the two points
            - Distance along x-axis
            - Distance along y-axis
            - Distance along z-axis
            - The index of the  fMRI cluster that is closest to the reference EEG/MEG point
    """
    # Convert points to numpy arrays for easy computation
    mri_points = np.array(mri_points)
    eg_point = np.array(eg_point)

    # Calculate differences in each dimension
    delta_x = mri_points[:,0] - eg_point[:,0]
    delta_y = mri_points[:,1] - eg_point[:,1]
    delta_z = mri_points[:,2] - eg_point[:,2]

    # Calculate the Euclidean distance
    euclidean_distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

   # Handle edge case when no valid distances are available
    if np.isnan(np.array(euclidean_distance, dtype=float)).all():
        return {
            'dist': np.nan,
            'distx': np.nan,
            'disty': np.nan,
            'distz': np.nan,
            'min_index': np.nan
        }

    # Find the index of the minimum distance
    min_dist_index = np.nanargmin(euclidean_distance)
    
    return {
        'dist': float(round(euclidean_distance[min_dist_index], 2)),
        'distx': float(round(delta_x[min_dist_index], 2)),
        'disty': float(round(delta_y[min_dist_index], 2)),
        'distz': float(round(delta_z[min_dist_index], 2)),
        'min_index': fmri_comparison_clusters[min_dist_index],
    }

def found_cog(cluster_map, cog_index=[1]):
    """
    Calculate the Center of Gravity (COG) for specified clusters in a 3D voxel map.

    Parameters:
    - cluster_map: A NIfTI image object or similar with a method `get_fdata()`
      that returns a 3D numpy array where the clusters are labeled with integers.
    - cog_index: A list of integers where each integer corresponds to a specific
      cluster label in the cluster_map. Default is [1].

    Returns:
    - A numpy array of shape (len(cog_index), 3) where each row corresponds to the COG
      of a cluster, with columns for the x, y, and z coordinates.
    """
    
    # Get the cluster map data
    cluster_data = cluster_map.get_fdata()

    # Initialize list to store COG coordinates for each cluster
    COG = []

    for idx in cog_index:
        # Get the coordinates of the voxels activated by the cluster
        cluster_voxels = np.where(cluster_data == idx)
        
        # Check if there are any voxels in the cluster
        if len(cluster_voxels[0]) == 0:
            # If no voxels found for this cluster index, append NaNs or zeroes
            COG.append([np.nan, np.nan, np.nan])
            continue

        # Calculate the COG for each dimension
        COGx = np.mean(cluster_voxels[0])
        COGy = np.mean(cluster_voxels[1])
        COGz = np.mean(cluster_voxels[2])

        # Pass in world coordinate
        [COGx, COGy, COGz] = nib.affines.apply_affine(cluster_map.affine, [COGx, COGy, COGz])

        # Append the COG coordinates to the list
        COG.append([COGx, COGy, COGz])

    # Convert the list to a numpy array and return
    return np.array(COG)

def fill_sub_df(df_sub_, task, condi, tp, modal_data, comparison_data, type_analysis, fmri_comparison_clusters):
    """
    Fills a DataFrame with task-specific data.

    Parameters:
    - df_sub_ (pd.DataFrame): The DataFrame to be filled. It must have a MultiIndex for columns.
    - task (str): The task identifier.
    - condi (str): The condition identifier.
    - tp (str): The time point index.
    - modal_data (list of lists): Contains modality-specific data. For 'POA' and 'COG', it's a list with:
        - fMRI coordinates (list of lists of floats)
        - EEG data (list of lists of floats)
        - (Optional) MEG data (list of lists of floats)
    - comparison_data (list of dicts or lists): Contains comparison metrics. For 'POA' and 'COG', it's a list with:
        - EEG distance metrics (dict with keys 'dist', 'distx', 'disty', 'distz')
        - (Optional) MEG distance metrics (dict with keys 'dist', 'distx', 'disty', 'distz')
    - type_analysis (str): Type of analysis, should be one of 'POA', 'COG', or 'Overlap'.
    - fmri_comparison_clusters (list): the fmri cluster to which the eeg and meg first cluster will be compared to


    Returns:
    - pd.DataFrame: The updated DataFrame with the new row filled with the provided data.
    
    Raises:
    - ValueError: If `type_analysis` is not one of the valid types ('POA', 'COG', 'Overlap').
    """

    # Validate the type_analysis argument
    valid_type_analysis = ['POA', 'COG', 'Overlap']
    if type_analysis not in valid_type_analysis:
        raise ValueError(f"type_analysis should be one of {valid_type_analysis}.")
        
    # Create a new row with NaN values for each column
    df_sub_.loc[len(df_sub_.index)] = [None] * (len(df_sub_.columns))

    # Get the index of the newly added row
    row_idx = len(df_sub_.index) - 1

    # Fill in the task-related information
    df_sub_.loc[row_idx, ('Info', 'task')] = task
    df_sub_.loc[row_idx, ('Info', 'condition')] = condi
    df_sub_.loc[row_idx, ('Info', 'tpindex')] = tp

    # Fill modality and comparison data based on type_analysis
    if type_analysis in {'POA', 'COG'}:
        # Update fMRI index and coordinates
        fMRI_index_eeg = comparison_data[0]['min_index']
        df_sub_.loc[row_idx, ('Info', 'fmridxEeg')] = fMRI_index_eeg 
        fMRI_index_meg = comparison_data[1]['min_index']
        df_sub_.loc[row_idx, ('Info', 'fmridxMeg')] = fMRI_index_meg 

        # Fill fMRI coords
        fmri_coords_eeg = modal_data[0][int(fmri_comparison_clusters.index(fMRI_index_eeg))] if not pd.isna(fMRI_index_eeg) else (np.nan, np.nan, np.nan)
        df_sub_.loc[row_idx, (type_analysis, 'fmrieeg')] = tuple(round(value, 2) for value in fmri_coords_eeg)
        fmri_coords_meg = modal_data[0][int(fmri_comparison_clusters.index(fMRI_index_meg))] if not pd.isna(fMRI_index_meg) else (np.nan, np.nan, np.nan)
        df_sub_.loc[row_idx, (type_analysis, 'fmrimeg')] = tuple(round(value, 2) for value in fmri_coords_meg)
        
        # Fill EEG data
        eeg_coords = modal_data[1][0] if not len(modal_data[1]) == 0 else (np.nan, np.nan, np.nan)
        df_sub_.loc[row_idx, (type_analysis, 'eeg')] = tuple(float(round(value, 2)) for value in eeg_coords)

        # Fill MEG data
        meg_coords = modal_data[2][0] if not len(modal_data[2]) == 0 else (np.nan, np.nan, np.nan)
        df_sub_.loc[row_idx, (type_analysis, 'meg')] = tuple(float(round(value, 2)) for value in meg_coords)        

         # Fill EEG distance metrics
        eeg_dist = comparison_data[0]
        df_sub_.loc[row_idx, ('DistEeg', 'x')] = eeg_dist['distx']
        df_sub_.loc[row_idx, ('DistEeg', 'y')] =  eeg_dist['disty']
        df_sub_.loc[row_idx, ('DistEeg', 'z')] = eeg_dist['distz']
        df_sub_.loc[row_idx, ('DistEeg', 'norm')] = eeg_dist['dist']

        # Fill MEG distance metrics
        meg_dist = comparison_data[1]
        df_sub_.loc[row_idx, ('DistMeg', 'x')] = meg_dist['distx']
        df_sub_.loc[row_idx, ('DistMeg', 'y')] =  meg_dist['disty']
        df_sub_.loc[row_idx, ('DistMeg', 'z')] = meg_dist['distz']
        df_sub_.loc[row_idx, ('DistMeg', 'norm')] = meg_dist['dist']

    elif type_analysis == 'Overlap':
        # Update fMRI index and cluster sizes
        fMRI_index_eeg = comparison_data[0]['cluster_index']
        df_sub_.loc[row_idx, ('Info', 'fmridxEeg')] = fMRI_index_eeg 
        fMRI_index_meg = comparison_data[1]['cluster_index']
        df_sub_.loc[row_idx, ('Info', 'fmridxMeg')] = fMRI_index_meg 

        # Fill fMRI data
        cluster_size_fmrieeg = modal_data[0][int(fmri_comparison_clusters.index(fMRI_index_eeg))] if not pd.isna(fMRI_index_eeg) else np.nan
        df_sub_.loc[row_idx, ('ClusterSizemm3', 'fmrieeg')] = cluster_size_fmrieeg
        cluster_size_fmrimeg = modal_data[0][int(fmri_comparison_clusters.index(fMRI_index_meg))] if not pd.isna(fMRI_index_meg) else np.nan
        df_sub_.loc[row_idx, ('ClusterSizemm3', 'fmrimeg')] = cluster_size_fmrimeg

        # Fill EEG cluster size
        df_sub_.loc[row_idx, ('ClusterSizemm3', 'eeg')] = modal_data[1] if len(modal_data[1]) !=0 else np.nan

        # Fill MEG cluster size
        df_sub_.loc[row_idx, ('ClusterSizemm3', 'meg')] = modal_data[2] if len(modal_data[2]) !=0 else np.nan

        # Fill EEG overlap metrics
        df_sub_.loc[row_idx, ('OverlapEeg', 'JaccardIdx')] = comparison_data[0]['jaccard_index']

        # Fill MEG overlap metrics
        df_sub_.loc[row_idx, ('OverlapMeg', 'JaccardIdx')] = comparison_data[1]['jaccard_index']

    return df_sub_

     

def fill_all_sub_df(df_all_sub_, sub, task, condi, tp, modal_data, comparison_data, type_analysis, fmri_comparison_clusters):
    """
    Fills a DataFrame with task-specific data.

    Parameters:
    - df_all_sub_ (pd.DataFrame): The DataFrame to be filled. It must have a MultiIndex for columns.
    - sub (str): The subject data identifier
    - task (str): The task identifier.
    - condi (str): The condition identifier.
    - tp (str): The time point index.
    - modal_data (list of lists): Contains modality-specific data. For 'POA' and 'COG', it's a list with:
        - fMRI coordinates (list of lists of floats)
        - EEG data (list of lists of floats)
        - (Optional) MEG data (list of lists of floats)
    - comparison_data (list of dicts or lists): Contains comparison metrics. For 'POA' and 'COG', it's a list with:
        - EEG distance metrics (dict with keys 'dist', 'distx', 'disty', 'distz')
        - (Optional) MEG distance metrics (dict with keys 'dist', 'distx', 'disty', 'distz')
    - type_analysis (str): Type of analysis, should be one of 'POA', 'COG', or 'Overlap'.
    - fmri_comparison_clusters (list): the fmri cluster to which the eeg and meg first cluster will be compared to

    Returns:
    - pd.DataFrame: The updated DataFrame with the new row filled with the provided data.
    
    Raises:
    - ValueError: If `type_analysis` is not one of the valid types ('POA', 'COG', 'Overlap').
    """

    # Validate the type_analysis argument
    valid_type_analysis = ['POA', 'COG', 'Overlap']
    if type_analysis not in valid_type_analysis:
        raise ValueError(f"type_analysis should be one of {valid_type_analysis}.")
        
    # Create a new row with NaN values for each column
    df_all_sub_.loc[len(df_all_sub_.index)] = [None] * (len(df_all_sub_.columns))

    # Get the index of the newly added row
    row_idx = len(df_all_sub_.index) - 1

    # Fill in the task-related information
    df_all_sub_.loc[row_idx, ('Info', 'SubjectName')] = sub
    df_all_sub_.loc[row_idx, ('Info', 'task')] = task
    df_all_sub_.loc[row_idx, ('Info', 'condition')] = condi
    df_all_sub_.loc[row_idx, ('Info', 'tpindex')] = tp
    
    # Fill modality and comparison data based on type_analysis
    if type_analysis in {'POA', 'COG'}:
        # Update fMRI index and coordinates
        fMRI_index_eeg = comparison_data[0]['min_index']
        df_all_sub_.loc[row_idx, ('Info', 'fmridxEeg')] = fMRI_index_eeg 
        fMRI_index_meg = comparison_data[1]['min_index']
        df_all_sub_.loc[row_idx, ('Info', 'fmridxMeg')] = fMRI_index_meg 

        # Fill fMRI coords
        fmri_coords_eeg = modal_data[0][int(fmri_comparison_clusters.index(fMRI_index_eeg))] if not pd.isna(fMRI_index_eeg) else (np.nan, np.nan, np.nan)
        df_all_sub_.loc[row_idx, (type_analysis, 'fmrieeg')] = tuple(round(value, 2) for value in fmri_coords_eeg)
        fmri_coords_meg = modal_data[0][int(fmri_comparison_clusters.index(fMRI_index_meg))] if not pd.isna(fMRI_index_meg) else (np.nan, np.nan, np.nan)
        df_all_sub_.loc[row_idx, (type_analysis, 'fmrimeg')] = tuple(round(value, 2) for value in fmri_coords_meg)
        
        # Fill EEG data
        eeg_coords = modal_data[1][0] if not len(modal_data[1]) == 0 else (np.nan, np.nan, np.nan)
        df_all_sub_.loc[row_idx, (type_analysis, 'eeg')] = tuple(float(round(value, 2)) for value in eeg_coords)

        # Fill MEG data
        meg_coords = modal_data[2][0] if not len(modal_data[2]) == 0 else (np.nan, np.nan, np.nan)
        df_all_sub_.loc[row_idx, (type_analysis, 'meg')] = tuple(float(round(value, 2)) for value in meg_coords)        

         # Fill EEG distance metrics
        eeg_dist = comparison_data[0]
        df_all_sub_.loc[row_idx, ('DistEeg', 'x')] = eeg_dist['distx']
        df_all_sub_.loc[row_idx, ('DistEeg', 'y')] =  eeg_dist['disty']
        df_all_sub_.loc[row_idx, ('DistEeg', 'z')] = eeg_dist['distz']
        df_all_sub_.loc[row_idx, ('DistEeg', 'norm')] = eeg_dist['dist']

        # Fill MEG distance metrics
        meg_dist = comparison_data[1]
        df_all_sub_.loc[row_idx, ('DistMeg', 'x')] = meg_dist['distx']
        df_all_sub_.loc[row_idx, ('DistMeg', 'y')] =  meg_dist['disty']
        df_all_sub_.loc[row_idx, ('DistMeg', 'z')] = meg_dist['distz']
        df_all_sub_.loc[row_idx, ('DistMeg', 'norm')] = meg_dist['dist']

    elif type_analysis == 'Overlap':
        # Update fMRI index and cluster sizes
        fMRI_index_eeg = comparison_data[0]['cluster_index']
        df_all_sub_.loc[row_idx, ('Info', 'fmridxEeg')] = fMRI_index_eeg 
        fMRI_index_meg = comparison_data[1]['cluster_index']
        df_all_sub_.loc[row_idx, ('Info', 'fmridxMeg')] = fMRI_index_meg 

        # Fill fMRI data
        cluster_size_fmrieeg = modal_data[0][int(fmri_comparison_clusters.index(fMRI_index_eeg))] if not pd.isna(fMRI_index_eeg) else np.nan
        df_all_sub_.loc[row_idx, ('ClusterSizemm3', 'fmrieeg')] = cluster_size_fmrieeg
        cluster_size_fmrimeg = modal_data[0][int(fmri_comparison_clusters.index(fMRI_index_meg))] if not pd.isna(fMRI_index_meg) else np.nan
        df_all_sub_.loc[row_idx, ('ClusterSizemm3', 'fmrimeg')] = cluster_size_fmrimeg

        # Fill EEG cluster size
        df_all_sub_.loc[row_idx, ('ClusterSizemm3', 'eeg')] = modal_data[1] if len(modal_data[1]) !=0 else np.nan

        # Fill MEG cluster size
        df_all_sub_.loc[row_idx, ('ClusterSizemm3', 'meg')] = modal_data[2] if len(modal_data[2]) !=0 else np.nan

        # Fill EEG overlap metrics
        df_all_sub_.loc[row_idx, ('OverlapEeg', 'JaccardIdx')] = comparison_data[0]['jaccard_index']

        # Fill MEG overlap metrics
        df_all_sub_.loc[row_idx, ('OverlapMeg', 'JaccardIdx')] = comparison_data[1]['jaccard_index']

    return df_all_sub_