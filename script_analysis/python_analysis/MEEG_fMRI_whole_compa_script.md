### The `MEEG_fMRI_whole_compa_script.py` script

This script orchestrates all the necessary steps in the analysis pipeline. Before running it, ensure that you **update the `config.py` file with the correct paths for your system**. Additionally, `config.py` includes various other parameters that can be updated, which are detailed in `config.txt`.

After updating `config.py`, execute `MEEG_fMRI_whole_compa_script.py`.

You can launch it using the following command:

```python
ipython path/to/MEEG_fMRI_whole_compa_script.py
```

The script will display the parameters defined in `config.py` and prompt you to confirm their correctness. It will then present the available sections for execution:

Available sections:
1. **Preparation of fMRI Data**: Processes and thresholds fMRI contrast data for further analysis.
2. **Preparation of EEG Data**: Interpolate source estimates to match the resolution of the anatomical MRI source image, resample to the fMRI resolution, and then apply a threshold to retain the top alpha percentile of the values.
3. **Preparation of MEG Data**: Convert source estimates files to NIfTI format, applying interpolation as needed. Resample the data to match the fMRI resolution, and then apply a threshold to retain only the top alpha percentile of the values.
4. **Generates Regions of Interest (ROIs) in the subject's native space**: Map a defined ATLAS into a specific subject space and create ROIs for various tasks.
5. **Generation of DataFrames for Comparing EEG and MEG Modalities to fMRI**: For each subject, task, and condition, this process calculates the distance between the Peak of Activation (POA) and the weighted Center of Gravity of defined fMRI clusters with the corresponding EEG and MEG clusters. It identifies and stores the clusters with the closest distances and computes the overlap between these closest clusters. The results are then organized into DataFrames—one for each subject and a summary DataFrame encompassing all subjects.
6. **Calculation of Mean and Standard Deviation for POA and COG Analyses**: For both modalities, the mean and standard deviation will be computed for various combinations of time points and conditions.
7. **Comparison of dMEG and dEEG Means to Zero**: Study if mean of dMEG (resp. dEEG) differ from zero.
8. **Comparison of Variability Between dEEG and dMEG**: Study if the spreads of dMEG and dEEG are different.

You can choose to run all sections or select specific ones based on your needs. Follow the instructions provided by the script to make your selections. Once you’ve chosen the sections, the script will proceed without further interruption.

Additionally, you will find other directories such as `data_preparation_notebooks` and `stats_notebooks`, which contain detailed notebooks for running different sections with additional explanations.

#### Detailed Steps for Each Section:

1. **Preparation of fMRI Data**:
   - This section prepares fMRI data for comparison by calling `fmri_data_preparation()`, which uses `mri_data_thresholding` from `mri_utils.py`. It processes the z-scores of fMRI contrasts to obtain thresholded z-scores. Parameters include LOCAL_DIR (the dataset_bids path), SUBJECTS, TASKS_CONDITIONS, logger, and plotting options.
   - Steps include loading the subject's T1 image, extracting the brain mask, processing or loading the z-score map, applying the brain mask, thresholding, and optionally plotting results.
   - For more details, refer to `data_preparation_notebooks/MRI_contrast.ipynb`, which provides explanations and details about the `mri_data_thresholding` function.

2. **Preparation of EEG Data**:
   - This section prepares the EEG data for comparison by invoking the `eeg_data_preparation()` function, which utilizes `eeg_data_stc2nifti` from `eeg_utils.py`. The function interpolates source estimates, obtained using the selected algorithm, onto the anatomical MRI source image. These estimates are then resampled to match the fMRI resolution and thresholded to retain the top alpha percentile.
   - The preparation involves several steps:
     1. Loading the source space for each subject, along with the corresponding T1 image and the brain mask resampled to the fMRI space.
     2. Load time windows for each task based on the time course.
     3. For each condition, loading the source estimates and identifying the time point with the maximum source estimate within each window.
     4. Interpolating the source estimates at this time point to match the MRI anatomical resolution.
     5. Resampling the interpolated data to the fMRI space using the brain mask, applying the mask to the resampled image.
     6. Thresholding the result to retain only the top alpha percentile of activation values, with a specified minimum cluster size.
   - For further details, refer to `data_preparation_notebooks/EEG_transform.ipynb`, which provides in-depth explanations and information about the `eeg_data_stc2nifti` function.
   - The function parameters include `LOCAL_DIR` (directory where BIDS data are saved), `logger`, `SUBJECTS`, `TASKS_CONDITIONS`, `EEG_TASKS_TP` (which defines the time windows for each task), `THRESHOLD_PARAMS`, and `EEG_STC_ALGO`.

3. **Preparation of MEG Data**: 
   - This section prepares MEG data for comparison by calling the function `meg_data_preparation()`, which utilizes `meg_data_stc2nifti` from `meg_utils.py`. The function converts source estimate files to NIfTI format, applying interpolation depending on `MRI_RESOLUTION`. It then resamples the data to match the fMRI resolution and applies a threshold to retain only the top alpha percentile of the values.
   - The preparation process involves several steps for each subject:
     1. **Load Data**: Load the forward model data, the brain mask resampled to the fMRI space, and optionally the T1w MRI.
     2. **Time Windows**: Load the time windows for each task based on the time course.
     3. **Process Conditions**: For each condition, load the source estimates (STC) file and identify the time point with the maximum source estimate within each time window.
     4. **Interpolation**: Depending on the value of `MRI_RESOLUTION`, either interpolate the STC data to the anatomical MRI resolution or save it directly in a downsampled resolution of (46, 60, 49).
     5. **Resampling and Masking**: Resample the interpolated data to the fMRI space using the brain mask. If `DO_BRAIN_MASKING` is enabled, apply the brain mask.
     6. **Thresholding**: Apply a threshold to retain only the top alpha percentile of activation values, with a specified minimum cluster size.
   - For more detailed information, refer to `data_preparation_notebooks/MEG_transform.ipynb`, which provides comprehensive explanations and details about the `meg_data_stc2nifti` function.
   - The function parameters include `LOCAL_DIR` (directory where BIDS data are saved), `logger`, `SUBJECTS`, `TASKS_CONDITIONS`, `MEG_TASKS_TP` (defines time windows for each task), `THRESHOLD_PARAMS`, `POS` (STC resolution), `MRI_RESOLUTION`, `DO_BRAIN_MASKING`, and `PLOTTING`.


4. **Generation of Regions of Interest (ROIs) in the Subject's Native Space**

- This section generates ROIs for each subject by mapping a defined ATLAS into the subject's native space. The function `generation_subs_rois()` performs this task, creating ROIs for various tasks such as 'aud', 'feet', 'fing', and 'vis'. The available atlases for ROI generation include the HOCPAL Atlases (th0 and th25), Schaefer2018 Atlases (7 and 17 parcels), and the AAL Atlas.

- The `generation_subs_rois()` function calls `create_subject_rois()` from `roi_utils.py`. This function requires several parameters, including the `subject`, the `ATLAS` (defined in `config.py`), and `LOCAL_DIR` (the directory where the BIDS data is saved). Additionally, you need to specify `PATH_TO_TPFLOW` (the path to TemplateFlow), `logger`, and `PLOTTING` (a boolean flag indicating whether to generate plots of the ROIs for each subject).

- To use the Schaefer2018 and HOCPAL atlases, you need the MNI152NLin6Asym template. Install TemplateFlow and download the template using the following commands:

  ```python
  pip install templateflow
  from templateflow import api as tflow
  tflow.get('MNI152NLin6Asym')
  ```

- For more details on the ROI generation process, refer to `data_preparation_notebooks/ROIs_handling.ipynb`, which provides further explanations and details about the `create_subject_rois()` function from `roi_utils.py`.
 
   **Note:** 
   - In `roi_utils.py`, the `ATLAS_REGIONS` variable defines the specific parcels for each atlas and task used to create the ROIs. If the predefined ROIs do not meet your requirements, you can update this variable to add or remove parcel names. 

5. **Generation of DataFrames for Comparing EEG and MEG Modalities to fMRI**:
   - This section facilitates the comparison between different modalities by invoking the `generation_comparison_modalities_df()` function, which uses `comparison_EEG_MEG_fMRI_df_fill` from `generation_compa_df_utils.py`. For each subject, task, and condition, it computes the distance between the Peak of Activation (POA) and the weighted Center of Gravity (COG) of fMRI clusters with corresponding EEG and MEG clusters. It identifies and records the clusters with the closest distances and calculates the overlap between these closest clusters. The results are then compiled into DataFrames—one per subject and one summarizing all subjects.
   - The comparison process involves the following steps for each subject and task:
     1. Load the fMRI statistical maps, as well as the EEG and MEG statistical maps for the specified time points.
     2. If `ROI_MASKING` is enabled, apply subject-specific Region of Interest (ROI) masks to the statistical maps.
     3. For each condition, extract cluster tables to obtain the POA location of the first cluster for EEG and MEG, and the chosen clusterd for fMRI. Obtain the cluster statistical maps as well.
     4. Calculate the distance between the MEG/EEG POA and each fMRI POA, keeping only the closest match for each time point.
     5. Compute the weighted Center of Gravity (COG) for the clusters of interest and again compute distances as in step 3.
     6. Determine the overlap between the EEG/MEG first cluster and the cluster found as closest with the COG.
     7. Save the results in a DataFrame.
     8. Save the results with one DataFrame per subject and another summarizing the information across all subjects.
   - For further details, refer to `data_preparation_notebooks/Generation_df_comparison.ipynb`, which provides in-depth explanations and information about the `comparison_EEG_MEG_fMRI_df_fill` function.
   - Function parameters include `sub_derivatives_outdir` (the directory for subject derivatives), the subject identifier, the task, task conditions, general and subject-specific DataFrames, fMRI correction parameters, EEG and MEG correction parameters, `EEG_STC_ALGO`, `POS` (the STC resolution), the logger, `FMRI_COMPARISON_CLUSTERS`, `DO_BRAIN_MASKING`, `ROI_MASKING` and `ATLAS`.

   **Note:**
   - When applying `ROI_MASKING`, it was observed that activations were often not detected. Consider performing the masking before thresholding to potentially improve accuracy.
   - There may also be issues with how the ROIs are defined, which could affect the results.

6. **Calculation of Mean and Standard Deviation for POA and COG Analyses**:

  - This section computes the mean and standard deviation for various combinations of time points and conditions for both modalities, using the general DataFrames obtained in the previous section. It utilizes the `mean_std_computation_step()` function, which in turn calls the `compute_mean_std` function from `stats_utils.py`.
  
  - The `mean_std_computation_step()` function requires two parameters:
    - `LOCAL_DIR`: The directory where the BIDS dataset is saved.
    - `analysis_type`: The type of analysis for which you want to compute the mean and standard deviation (either POA or COG).

  - The function calculates the mean and standard deviation for the following variables: `Dist_x`, `Dist_y`, `Dist_z`, and `Dist_norm`. It performs these calculations for:
    - All conditions and time points combined.
    - Conditions separately and time points combined.
    - Time points separately and conditions combined.
    - Each time point and condition combination.

  - For more detailed explanations and information about the `compute_mean_std` function, refer to `stats_notebooks/Mean_Std_Computation.ipynb`.

7. **Comparison of dMEG and dEEG Means to Zero**:

   - This section tests whether the means of dMEG and dEEG differ from zero using a multivariate non-parametric location test based on the method described by Oja and Randles (2004) ([Link to article](http://www.jstor.org/stable/4144430)). The `mean_stats_analysis()` function is used, which calls the `mean_diff_zero` function from `stats_utils.py`. This function relies on the R script `diff_zero.R` for its computations.

   - Required parameters for the function include: the R working directory (`R_WORKING_DIRECTORY`), the path to input XLSX files, the path to the Rscript executable (`RSCRIPT_EXECUTABLE`), the analysis type (COG or POA), the number of permutations (`nb_permu`), the hypothesized location value (`null_value`, set to `MEAN_COMPA_VALUE`), and the method of analysis (`MEAN_TYPE_ANALYSIS`, which can be 'sign' or 'rank').

   - **Procedure**:
     1. Process MEG and EEG data separately.
     2. Use the `sr.loc.test` function from the `SpatialNP` R package on variables Distx, Disty, and Distz.
     3. This multivariate test evaluates the location of one or more samples based on spatial signs or ranks. For a single sample, it tests whether the sample location equals a specified value (in this case, the origin (0,0,0)). For multiple samples, it tests whether all samples share the same location.
     4. Save the results in a new DataFrame at `derivatives/results/{analysis_type}/all_subjects_analysis-{analysis_type}_modality_comparison_analysis-location_R.csv`.

   - For detailed explanations and further information, refer to the `stats_notebooks/Location_statistic.ipynb` notebook, which provides an in-depth look at the use of `compute_mean_std` and the R process.

   **Note**:
   - If the dataset contains identical rows across all three variables, the function may hang indefinitely. To resolve this, uncomment lines 157, 158, 170, and 171 in the `statistical_zero_analysis_utils.R` script. This ensures that each row in the DataFrame is unique before the statistical computation begins.

8. **Comparison of Variability Between dEEG and dMEG**:

   - This section assesses whether the variability of dMEG and dEEG differs using a non-parametric multivariate analogue of Levene's test for homogeneity of variances, as described by Anderson (2006) ([DOI: 10.1111/j.1541-0420.2005.00440.x](https://doi.org/10.1111/j.1541-0420.2005.00440.x)). The `dispersion_stats_analysis()` function is called, which then invokes either the `dispersion_analysis_R` function from `stats_utils.py` or the `dispersion_analysis_python` function, depending on the `DISPERSION_SCRIPT` setting. Both scripts perform the same analysis but use different underlying methods: one utilizes the `compute_dispersion_analysis` function from `dispersion_stats_analysis_utils.py`, while the other uses the R script `dispersion_analysis.R`.

   - The required parameters include: the path to the XLSX files, the analysis type (POA or COG), the number of permutations, the distance metric, and for the R script, the path to the R script executable and working directory.

   - **Procedure**:
     1. Load the DataFrame.
     2. Provide separate rows for each modality (EEG and MEG) instead of combining them in a single row, and add a column labeled 'modality'.
     3. Compute the dispersion analysis using either the `permdisp` function from `scikit-bio` in Python or the `betadisper` function from the `vegan` package in R. This multivariate test evaluates whether the dispersion between groups is significant. The variables used are Dist_x, Disty, and Distz.
     4. If the results are significant, generate a PCA plot.
     5. Save the results in a new DataFrame at `derivatives/results/{analysis_type}/all_subjects_analysis-{analysis_type}_modality_comparison_analysis-dispersion_R/python.csv`.

   - For detailed explanations and further information, refer to the `stats_notebooks/Dispersion_statistic.ipynb` notebook, which provides an in-depth look at the use of `dispersion_analysis_python` and  `dispersion_analysis_R` and their processes.

   **Note**:
   - The R script is recommended for faster performance.

**Statistical Analyses Notes**:
   - Both types of statistical analyses are performed for:
     - All conditions and time points combined.
     - Conditions separately and time points combined.
     - Time points separately and conditions combined.
     - Each time point and condition combination.

---
