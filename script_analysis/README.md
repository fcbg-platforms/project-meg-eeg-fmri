## Setup Instructions

Before running the script `MEEG_fMRI_whole_compa_script.py`, please ensure you have updated the necessary paths in `config.py` to match your system's configuration.

### R Installation

To perform the statistical analysis, R must be installed on your computer. Additionally, `Rscript` might be useful. You can download R from:

- **For Windows:** [R for Windows](https://cran.r-project.org/bin/windows/base/)
- **For macOS:** [R for macOS](https://cran.r-project.org/bin/macosx/)
- **For Linux:** [R for Linux](https://cran.r-project.org/bin/linux/)

We also recommend installing RStudio for a more user-friendly interface:

- **RStudio:** [Download RStudio](https://posit.co/downloads/)

In the `/R_analysis/` directory, you will find the script `r_scripts/requirements.R`. This script will automatically download all the required R packages needed for the statistical analysis. Don't forget to run it.

### Python Environment Setup

The Python environment required for running the functions is specified in the `python_analysis/requirements.txt` file. To install the necessary packages, you can use either `conda` or `pip`:

- **Using Conda:**

  ```cmd
  conda install --file requirements.txt
  ```

- **Using Pip:**

  ```bash
  pip install -r requirements.txt
  ```

The code has been developed and tested with Python 3.12.5. Ensure you are using this version or a compatible one to avoid any compatibility issues.

### Data Info Directory

In this repository, you will find a directory named `data_info`, which contains five Excel files named in the format: `Corresp_SPM_con_{task}.xlsx` and `Corresp_SPM_regressors_{task}.xlsx`. These files are crucial if new subjects outside of the predefined list (`['sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17']`) need to be studied.

These files are crucial for mapping SPM contrast numbers (e.g., `spmT_0001`) generated after running the GLM to their corresponding contrast names (e.g., task: aud, contrast: sine) when saving in BIDS format. They are also necessary for saving `con_xxxx` and `beta_xxxx` files in BIDS format.

The Excel files are used in the script `mri2bids.py`.

### The `python_analysis` Directory

In this directory, you'll find the main script, `MEEG_fMRI_whole_compa_script.py`, which orchestrates all the necessary steps in the analysis pipeline. Detailed explanations for this script are provided in `MEEG_fMRI_whole_compa_script.md`. 

The `config.py` file contains all the variables needed to run the various scripts and notebooks. Be sure to update the paths in this file to match your local system. For further details about these variables, refer to `config.md`.

The `mri2bids.py` script converts `MRI_analyses` data in the subject space to BIDS format. Details about this process can be found in `mri2bids.md`.

Within the `data_preparation_notebooks` and `stats_notebooks` directories, you'll find various notebooks that explain the different steps of the `MEEG_fMRI_whole_compa_script.py`. Additionally, the `utils` directory contains the necessary functions to support the script.

### The `r_scripts` Directory

This directory contains R scripts used for statistical analysis. The `diff_zero.R` and `dispersion_analysis.R` scripts perform non-parametric multivariate location tests and Levene's test analogs for homogeneity of variances, respectively. These scripts are called by the `MEEG_fMRI_whole_compa_script.py` Python script. The utility functions for these analyses are provided in `statistical_zero_analysis_utils.R` and `statistical_dispersion_analysis_utils.R`. 