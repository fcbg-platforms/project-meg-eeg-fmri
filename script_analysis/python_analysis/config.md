### Configuration Variables Description
This file describes the configuration variables used in `config.py`. 

#### General Settings
- **`LOCAL_DIR`**: `str`  
  Path to the `dataset_bids` directory.

- **`PATH_TO_MRI_DATA`**: `str`  
  Path to the MRI data output by SPM, typically found in the `MRI_analyses` directory.

- **`MRI_DATA_CON_INFO_DIR`**: `str`  
  Path to the `Corresp_SPM_con_{task}.xlsx` files, available for download from the `data_info` directory in the GitHub repository.

- **`PATH_TO_TPLFLOW`**: `str`  
  Path to the TemplateFlow `MNI152NLin6Asym` template. This template is required if you intend to create ROIs using the HOCPAL or Schaefer2018 atlases. To download this template, you can use the following command:
  ```python
  pip install templateflow
  from templateflow import api as tflow
  tflow.get('MNI152NLin6Asym')
  ```

- **`R_WORKING_DIRECTORY`**: `str`  
  Path to the directory containing R scripts downloaded from GitHub.

- **`RSCRIPT_EXECUTABLE`**: `str`  
  Path to the R script executable. You can locate it with:
  - **Windows:** `where rscript.exe`
  - **macOS/Linux:** `which rscript`
  
  If no path is returned, manually locate the R installation directory, usually found in `C:\Program Files\R\R-x.x.x\bin` on Windows.

- **`SUBJECTS`**: `List[str]`  
  List of subject identifiers in the format `['sub-xx']`, specifying the subjects to study.

#### Contrast Settings
- **`TASKS_CONDITIONS`**: `dict`  
  Dictionary mapping each task to its associated conditions. This dictionary is used across all modalities, so ensure the tasks and contrasts are consistent.

- **`EEG_TASKS_TP`**: `dict`  
  For EEG data preparation. Defines three time windows for each task in *frames*. For each condition and subject, the time point with the maximum activation is identified, and the NIfTI file is saved for that time point. Different time windows can be specified for each condition if they are not time-locked.

- **`MEG_TASKS_TP`**: `dict`  
  Similar to `EEG_TASKS_TP` but for MEG data. Defines three time windows in *seconds* per task, with NIfTI files saved for the time points of maximum activation.

- **`THRESHOLD_PARAMS`**: `dict`  
  Parameters for thresholding EEG and MEG NIfTI files:
  - **`alpha`**: Top alpha percentile to retain.
  - **`cluster_thres`**: Minimum cluster size allowed.

- **`CorrectionParam`**: `class`  
  Encapsulates correction parameters for fMRI data analysis:
  - **`self.fMRI_ALPHA`**: False positive rate after correction.
  - **`self.fMRI_CORRECTION_METHOD`**: Type of correction (`'fpr'`, `'fdr'`, `'bonferroni'`).
  - **`self.fMRI_CLUSTER_THRES`**: Minimum cluster size allowed.
  - **`self.fMRI_TWOSIDED`**: Whether to use two-sided correction (retain negative values or not).

- **`EEG_STC_ALGO`**: `str`
  The algorithm used to compute the eeg sources estimates.

- **`POS`**: `int`  
  Resolution of the source estimate in millimeters for MEG data.

- **`DO_BRAIN_MASKING`**: `bool`  
  If `True`, MEG data will be brain-masked before thresholding in the `meg_data_preparation()` function. Masking may be optional depending on interpolation and resampling needs.

- **`MRI_RESOLUTION`**: `bool`  
  If `True`, during interpolation from STC to NIfTI, the data will be resampled from the downsampled space (46, 60, 49) to the anatomical MRI space (256, 256, 256) before resampling to the fMRI space (112, 112, 66). If `False`, the data will only be resampled directly from (46, 60, 49) to (112, 112, 66).

- **`PLOTTING`**: `bool`  
  If `True`, plots will be generated for each subject's condition after thresholding during preparation sections.


- **`MEG_STC_ALGO`**: `str`
  The algorithm used to compute the meg source estimates.

#### ROIs Generation Parameters

- **`ATLAS`**: `str`
  Specifies the atlas used for generating Regions of Interest (ROIs). Available options are:
  - `'HOCPALth0`
  - `'HOCPALth25'`
  - `'Schaefer7'`
  - `'Schaefer17'`
  - `'AAL'`

  The selection of ROIs within each atlas is based on predefined parcels. Details about which parcels are included in each ROI for the different atlases can be found in the `roi_utils.py` file, in the ATLAS_REGIONS variable.

#### Comparison DataFrame Generation Parameters
- **`FMRI_COMPARISON_CLUSTERS`**: `list`
  A list containing the indices of the fMRI clusters that will be compared with the EEG and MEG clusters. If multiple fMRI clusters are present, only the closest one is saved in the comparison dataframe. This allows you to manually select the specific fMRI cluster to compare with the other modalities. The index corresponds to the order of the clusters based on their peak activation.

- **`ROI_MASKING`**: `bool`  
  Specifies whether to apply ROI-based masking to the fMRI, EEG, and MEG statistical maps. When set to `True`, the statistical maps will be masked before comparison using the ROIs associated with the current task, as defined by the atlas specified in the `ATLAS` parameter.

#### Statistical Analysis Parameters
- **`NB_PERMUTATIONS`**: `int`  
  Number of permutations for statistical analysis (both dispersion and location). Maximum of 999.

- **`DISPERISON_DIST_METRICS`**: `str`  
  Metric used to compute the distance matrix for dispersion analysis. Refer to [SciPy distance metrics](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) or [vegan package in R](https://search.r-project.org/CRAN/refmans/vegan/html/vegdist.html) for available metrics.

- **`MEAN_COMPA_VALUE`**: `list`  
  List of values to compare the mean against. Set to `None` to compare to zero.

- **`MEAN_TYPE_ANALYSIS`**: `str`  
  Type of analysis for location statistics. Options are `'sign'` or `'rank'`. For details, see: Oja, H., & Randles, R. H. (2004). Multivariate Nonparametric Tests. *Statistical Science, 19*(4), 598–605. [Link to article](http://www.jstor.org/stable/4144430).

- **`DISPERISON_SCRIPT`**: `str`  
  Choose the script for dispersion statistical analysis: `'R'` or `'Python'`. Select which script to use based on your preference.


