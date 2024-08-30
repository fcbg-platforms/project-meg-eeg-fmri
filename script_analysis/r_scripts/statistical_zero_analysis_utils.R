# Load the packages
library(SpatialNP)
library(dplyr)
library(tidyr)
library(stringr)

# Define the function to perform the multivariate sign test
perform_multivariate_sign_test <- function(data, nullvalue = NULL, cond = TRUE, cond.n = 1000, score = 'sign') {
  
  # Check if the sample size is valid
  if (nrow(data) == 0) {
    stop("The dataset is empty.")
  }
  
  # Perform the spatial sign test
  test_result <- sr.loc.test(
    X = data,
    Y = NULL,
    g = NULL,
    score = score,
    nullvalue = nullvalue,
    cond = cond,
    cond.n = cond.n,
    na.action = na.omit
  )
  
  # Combine the test result with sample size
  result_with_size <- list(
    test_result = test_result,
    sample_size = nrow(data),
    score = score
  )
  
  return(result_with_size)
}


# Define a function to prepare DataFrames for a specific modality
prepare_modality_df <- function(df, modality) {
  if (modality == "dEEG") {
    columns_to_keep <- c("Info_SubjectName", "Info_task", "Info_condition", "Info_tpindex",
                         "DistEeg_x", "DistEeg_y", "DistEeg_z")
  } else if (modality == "dMEG") {
    columns_to_keep <- c("Info_SubjectName", "Info_task", "Info_condition", "Info_tpindex",
                         "DistMeg_x", "DistMeg_y", "DistMeg_z")
  } else {
    stop("Modality must be either 'dEEG' or 'dMEG'.")
  }
  
  # Check if all required columns are present
  missing_columns <- setdiff(columns_to_keep, names(df))
  if (length(missing_columns) > 0) {
    stop(paste("Missing columns:", paste(missing_columns, collapse = ", ")))
  }
  
  # Prepare the modality-specific DataFrame
  df_modality <- df %>%
    select(all_of(columns_to_keep)) %>%
    mutate(modality = modality)
  
  return(df_modality)
}

# Define a function to clean column names
clean_column_names <- function(df) {
  names(df) <- gsub("eeg|meg|Eeg|Meg", "", names(df))
  names(df) <- gsub("_+", "_", names(df))
  names(df) <- gsub("_$", "", names(df))
  return(df)
}

# Define a function to return significance codes
significance_code <- function(p_value) {
  if (p_value <= 0.001) {
    return("***")
  } else if (p_value <= 0.01) {
    return("**")
  } else if (p_value <= 0.05) {
    return("*")
  } else if (p_value <= 0.1) {
    return(".")
  } else {
    return(" ")
  }
}

# Define a function to initialize the results DataFrame
def_location_results_df <- function() {
  results_location_analysis <- data.frame(
    modality = character(),
    condition = character(),
    tp = character(),
    method_name = character(),
    sample_size = numeric(),
    number_of_groups = numeric(),
    Comparison_Value = character(),
    q_2 = numeric(),
    p_value = character(),
    number_of_permutations = numeric(),
    stringsAsFactors = FALSE
  )
  return(results_location_analysis)
}

# Define the main function for dispersion analysis
# (Handling duplicate rows is essential in this function to avoid issues with the sr.test.loc function,
# which may produce incorrect results or errors when processing datasets with redundant rows)
compute_location_analysis <- function(df, results_location_analysis, modality, score = 'sign', nb_permutations = 1000, null_value = NULL, path_to_data = NULL) {
  
  add_results <- function(condition, tp, modality, location_test_results) {
    # Add the results to the DataFrame with significance codes
    new_row <- data.frame(
      modality = modality,
      condition = condition,
      tp = tp,
      method_name = location_test_results$score,
      q_2 = round(as.numeric(location_test_results$test_result$statistic), 2),
      sample_size = location_test_results$sample_size,
      number_of_groups = 1,
      Comparison_Value = location_test_results$test_result$null.value,
      p_value = sprintf("%.3f (%s)", location_test_results$test_result$p.value, significance_code(location_test_results$test_result$p.value)),
      number_of_permutations = as.numeric(location_test_results$test_result$parameter),
      stringsAsFactors = FALSE
    )
    
    # Combine with results DataFrame
    results_location_analysis <<- rbind(results_location_analysis, new_row)
  }
  
  # Compute dispersion for the entire dataset
  poa_df <- df %>%
    select(starts_with("Dist")) %>%
    drop_na()
  
  # Perform the multivariate location test
  location_test_result <- perform_multivariate_sign_test(poa_df, nullvalue = null_value, cond = TRUE, cond.n = nb_permutations, score = score)
  add_results("all", "all", modality, location_test_result)

  # Compute dispersion for each time point
  for (tp in unique(df$Info_tpindex)) {
    filtered_df <- df %>% filter(Info_tpindex == tp)
    poa_df <- filtered_df %>%
      select(starts_with("Dist")) %>%
      drop_na()
    
    # Perform the multivariate location test
    location_test_result <- perform_multivariate_sign_test(poa_df, nullvalue = null_value, cond = TRUE, cond.n = nb_permutations, score = score)
    add_results("all", tp, modality, location_test_result)

  }
  
  # Compute dispersion for each condition
  for (condi in unique(df$Info_condition)) {
    filtered_df <- df %>% filter(Info_condition == condi)
    poa_df <- filtered_df %>%
      select(starts_with("Dist")) %>%
      drop_na() # %>%
      # unique() # Because problem with duplicate row when not a lot of element
    
    # Perform the multivariate location test
    location_test_result <- perform_multivariate_sign_test(poa_df, nullvalue = null_value, cond = TRUE, cond.n = nb_permutations, score = score)
    add_results(condi, "all", modality, location_test_result)
    
    # Compute dispersion for each time point within each condition
    for (tp in unique(filtered_df$Info_tpindex)) {
      filtered_df_tp <- filtered_df %>% filter(Info_tpindex == tp)
      
      poa_df_tp <- filtered_df_tp %>%
        select(starts_with("Dist")) %>%
        drop_na() # %>%
        # unique() # Because problem with duplicate row when not a lot of element
      
      # Perform the multivariate location test
      location_test_result_tp <- perform_multivariate_sign_test(poa_df_tp, nullvalue = null_value, cond = TRUE, cond.n = nb_permutations, score = score)
      add_results(condi, tp, modality, location_test_result_tp)
    }
  }
  
  return(results_location_analysis)}
