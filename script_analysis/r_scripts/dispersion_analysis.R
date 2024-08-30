# Load necessary libraries
library(readr)     # For reading Excel files
library(dplyr)      # For data manipulation
library(vegan)      # For distance and dispersion analysis

# Define the default working directory and data path
default_working_directory <- "C://Users/lilia.ponselle/Documents/GitHub/Internship_MEEG_IRM/r_scripts/"
default_analysis_type <- "POA"
default_path_to_data <- paste0("I:/9008_CBT_HNP/MEG-EEG-fMRI/dataset_bids/derivatives/results/", default_analysis_type)
default_nb_permu <- 999
default_dist_metric <- 'euclidean'

# Function to parse named arguments
parse_args <- function(args) {
  parsed_args <- list()
  for (arg in args) {
    split_arg <- strsplit(arg, "=")[[1]]
    if (length(split_arg) == 2) {
      parsed_args[[split_arg[1]]] <- split_arg[2]
    }
  }
  return(parsed_args)
}

# Read command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Initialize variables with default values
working_directory <- default_working_directory
path_to_data <- default_path_to_data
analysis_type <- default_analysis_type
nb_permu <- default_nb_permu
dist_metric <- default_dist_metric

# Parse command-line arguments
parsed_args <- parse_args(args)

# Update variables based on parsed arguments
if (!is.null(parsed_args$working_directory)) {
  working_directory <- parsed_args$working_directory
}
if (!is.null(parsed_args$path_to_data)) {
  path_to_data <- parsed_args$path_to_data
}
if (!is.null(parsed_args$analysis_type)) {
  analysis_type <- parsed_args$analysis_type
}
if (!is.null(parsed_args$nb_permu)) {
  nb_permu <- as.numeric(parsed_args$nb_permu)
}
if (!is.null(parsed_args$dist_metric)) {
  dist_metric <- parsed_args$dist_metric
}

# Validate the analysis_type argument
if (!analysis_type %in% c("POA", "COG")) {
  stop("Error: analysis_type must be either 'POA' or 'COG'.")
}

if (nb_permu > 999) {
  warning('The maximum number of permutations is 999. The value of `nb_permu` has been capped at 999.')
  nb_permu <- 999
}

# Print the working directory and data path being used for confirmation
cat("Using working directory:", working_directory, "\n")
cat("Using path to data:", path_to_data, "\n")
cat("Analysis type:", analysis_type, "\n")
cat("Number of permutations:", nb_permu, "\n")
cat("Distance metric:", dist_metric, "\n")

# Set the working directory
setwd(working_directory)

# Source the script containing utility functions
source("statistical_dispersion_analysis_utils.R")

# Description:
# This script performs dispersion analysis on EEG and MEG data from an Excel file.
# It supports named command-line arguments for customization:
# - working_directory: The directory where the script will run and look for data files.
# - path_to_data: The directory containing the data and where results will be saved.
# - analysis_type: The analysis studied: POA or COG
# - nb_permu: Number of permutations for the dispersion analysis. Default is 999
# - dist_metric: Distance metric used in dispersion analysis. Default is 'euclidean'.

#
# The analysis includes:
# 1. Reshaping the data to combine EEG and MEG modalities and adding a modality column.
# 2. Performing dispersion analysis using a specified distance metric and number of permutations.
# 3. Saving the results to an Excel file.

# Parameters:
# nb_permu: Number of permutations for the dispersion analysis. Default is 999
# dist_metric: Distance metric used in dispersion analysis. Default is 'euclidean'.
# path_to_data: Path to the directory containing the data and where results will be saved. 
#               If not provided via command-line argument, defaults to "I:/9008_CBT_HNP/MEG-EEG-fMRI/dataset_bids/derivatives_2/".
# working_directory: Path to the directory containing the Rscripts. 
#               If not provided via command-line argument, defaults to "C://Users/lilia.ponselle/Documents/GitHub/Internship_MEEG_IRM/Analysis/R_analysis/".
# analysis_type: The analysis studied: POA or COG, defaults to 'POA'

# Define the path to the data
data_file_path <- file.path(path_to_data, paste0("all_subjects_analysis-", analysis_type, "_modality_comparison.csv"))

# Load DataFrame from the Excel file
data_df <- read_csv(data_file_path, show_col_types = FALSE)

# Optional: Display the first few rows of the DataFrame to verify its contents
# print(head(data_df))

# Reshape DataFrame to combine EEG and MEG modalities and add a modality column
combined_data_df <- reshape_dataframe(data_df)

# Optional: Display the first few rows of the reshaped DataFrame
# print(head(combined_data_df))

# Perform dispersion analysis on the reshaped DataFrame
results_dispersion_analysis <- compute_dispersion_analysis(
  combined_data_df,
  analysis_type = analysis_type,
  dist_metric = dist_metric, 
  nb_permutations = nb_permu, 
  path_to_data = path_to_data
)

# Display the results of the dispersion analysis
print(results_dispersion_analysis)

# Save the dispersion results to an Excel file
path_dispersion_results <- file.path(path_to_data, paste0("all_subjects_analysis-", analysis_type, "_modality_comparison_stats-dispersion_R.csv"))
write.csv(results_dispersion_analysis, path_dispersion_results, row.names = FALSE)

