# Load required libraries
library(readr)
library(dplyr)
require(magrittr)

# Define the default working directory and data path
default_working_directory <- "C://Users/lilia.ponselle/Documents/GitHub/Internship_MEEG_IRM/r_scripts/"
default_analysis_type <- "POA"
default_path_to_data <- paste0("I:/9008_CBT_HNP/MEG-EEG-fMRI/dataset_bids/derivatives/results/", default_analysis_type)
default_nb_permu <- 1000
default_null_value <- NULL
default_score <- 'sign'

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
null_value <- default_null_value
score <- default_score

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

# Convert "NULL" to NULL
if (!is.null(parsed_args$null_value)) {
  if (parsed_args$null_value == "NULL") {
    null_value <- NULL
  } else {
    null_value <- as.numeric(parsed_args$null_value)
  }
}

if (!is.null(parsed_args$score)) {
  score <- parsed_args$score
}

# Validate the analysis_type argument
if (!analysis_type %in% c("POA", "COG")) {
  stop("Error: analysis_type must be either 'POA' or 'COG'.")
}

# Validate the score argument
if (!score %in% c("sign", "rank")) {
  stop("Error: score must be either 'sign' or 'rank'.")
}

# Print the working directory and data path being used for confirmation
cat("Using working directory:", working_directory, "\n")
cat("Using path to data:", path_to_data, "\n")
cat("Analysis type:", analysis_type, "\n")
cat("Number of permutations:", nb_permu, "\n")
cat("Null value:", null_value, "\n")
cat("Scoring method:", score, "\n")

# Set the working directory
setwd(working_directory)

# Source the script containing utility functions
source("statistical_zero_analysis_utils.R")

# Description:
# This script tests if the mean of dMEG (or dEEG) variables differs from zero.
# The script requires adjustments for parameters such as the path to the data and the number of permutations.
# It reads the POA or COG data from an Excel file, processes it, performs statistical tests, and saves the results.
#
# Parameters:
# - working_directory: The directory where the script runs and looks for data files. Default is "C://Users/lilia.ponselle/Documents/GitHub/Internship_MEEG_IRM/Analysis/R_analysis/".
# - path_to_data: The directory containing the data and where results will be saved. Default is "I:/9008_CBT_HNP/MEG-EEG-fMRI/dataset_bids/derivatives_2/".
# - analysis_type: The analysis studied: POA or COG. Default is 'POA'.
# - nb_permu: Number of permutations for the location analysis. Default is 1000, capped at 999.
# - null_value: The value to which the variables are compared. Default is NULL.
# - score: The scoring method to be used in the analysis. Default is 'sign'.


# Define the path to the data
data_file_path <- file.path(path_to_data, paste0("all_subjects_analysis-", analysis_type, "_modality_comparison.csv"))

# Load DataFrame from the Excel file
data_df <- read_csv(data_file_path, show_col_types = FALSE)

# Optional: Display the first few rows of the DataFrame to verify its contents
# print(head(data_df))

results_location_analysis <- def_location_results_df()

# Define the results dataframe for location analysis
for (modality in c("dEEG", "dMEG")) { 
  
  # Select the data of interest for this modality
  modality_df <- prepare_modality_df(data_df, modality)
  modality_df <- clean_column_names(modality_df)
  
  # Optional: Display the first few rows of the DataFrame to verify its contents
  # print(head(modality_df))
  
  # Run the analysis
  results_location_analysis <- compute_location_analysis(df = modality_df, 
                                                         results_location_analysis = results_location_analysis, 
                                                         modality = modality, 
                                                         score = score,  # Adjust score parameter as needed
                                                         nb_permutations = nb_permu,
                                                         null_value = null_value, 
                                                         path_to_data = path_to_data)
}

# Display the results of the location analysis
print(results_location_analysis)

# Save the location results to an Excel file
path_location_results <- file.path(path_to_data, paste0("all_subjects_analysis-", analysis_type, "_modality_comparison_stats-location_R.csv"))
write.csv(results_location_analysis, path_location_results, row.names = FALSE)

