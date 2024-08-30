# Load necessary libraries
library(readxl)     # For reading Excel files
library(writexl)    # For writing Excel files
library(dplyr)      # For data manipulation
library(vegan)      # For distance and dispersion analysis
library(tidyr)      # For tidying data
library(ggplot2)    # For plotting
library(fs)         # For file system operations

# Define a function to prepare DataFrames for a specific modality
prepare_modality_df <- function(df, modality) {
  if (modality == "eeg") {
    columns_to_keep <- c("Info_SubjectName", "Info_task", "Info_condition", "Info_tpindex",
                         "Info_fmridxEeg", "DistEeg_x", "DistEeg_y", "DistEeg_z")
  } else if (modality == "meg") {
    columns_to_keep <- c("Info_SubjectName", "Info_task", "Info_condition", "Info_tpindex",
                         "Info_fmridxMeg", "DistMeg_x", "DistMeg_y", "DistMeg_z")
  } else {
    stop("Modality must be either 'eeg' or 'meg'.")
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

# Main function to reshape the DataFrame
reshape_dataframe <- function(df) {
  df_eeg <- prepare_modality_df(df, "eeg")
  df_meg <- prepare_modality_df(df, "meg")
  
  df_eeg_clean <- clean_column_names(df_eeg)
  df_meg_clean <- clean_column_names(df_meg)
  
  df_combined <- bind_rows(df_eeg_clean, df_meg_clean)
  
  return(df_combined)
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

# Define the main function for dispersion analysis
compute_dispersion_analysis <- function(df, analysis_type = 'POA', dist_metric = 'euclidean', nb_permutations = 999, path_to_data = NULL) {
  # Initialize results DataFrame
  results_dispersion_analysis <- data.frame(
    condition = character(),
    tp = character(),
    method_name = character(),
    sample_size = numeric(),
    number_of_groups = numeric(),
    Df = character(),
    Sum_Sq = character(),
    Mean_Sq = numeric(),
    test_statistic = numeric(),
    p_value = character(),
    number_of_permutations = numeric(),
    stringsAsFactors = FALSE
  )
  
  add_results <- function(condition, tp, permutest_results, betadisper_results) {
    # Add the results to the DataFrame with significance codes
    new_row <- data.frame(
      condition = condition,
      tp = tp,
      method_name = "betadisper",
      test_statistic_name = "Beta dispersion",
      sample_size = length(betadisper_results$distances),
      number_of_groups = length(unique(df$modality)),
      Df = permutest_results$tab$`Df`[1],
      Sum_Sq = permutest_results$tab$`Sum Sq`[1],
      Mean_Sq = permutest_results$tab$`Mean Sq`[1],
      test_statistic = permutest_results$tab$F[1],
      p_value = sprintf("%.3f (%s)", permutest_results$tab$`Pr(>F)`[1], significance_code(permutest_results$tab$`Pr(>F)`[1])),
      number_of_permutations = permutest_results$tab$`N.Perm`[1],
      stringsAsFactors = FALSE
    )
    
    # Combine with results DataFrame
    results_dispersion_analysis <<- rbind(results_dispersion_analysis, new_row)
  }
  
  
  # Compute dispersion for the entire dataset
  all_poa_df <- df %>%
    select(Dist_x, Dist_y, Dist_z) %>%
    drop_na()
  
  modalities <- df %>% select(modality)
  dist_matrix <- vegdist(all_poa_df, method = dist_metric)
  betadisper_results <- betadisper(dist_matrix, modalities$modality)
  permutest_results = permutest(betadisper_results, pairwise = TRUE, permutations = nb_permutations)
  add_results("all", "all", permutest_results, betadisper_results)
  
  # Plot if significant results
  if (permutest_results$tab$`Pr(>F)`[1] <= 0.05) {
    title <- sprintf("%s Dispersion Test: tp = %s, condi = %s, p_value = %.3f", analysis_type, 'all', 'all', permutest_results$tab$`Pr(>F)`[1])
    pca_plot <- plot(betadisper_results, main = title)
    
    if (!is.null(path_to_data)) {
      save_pca_plot(plot_func = function() plot(betadisper_results, main = title),
                    path_to_data = path_to_data, analysis_type = analysis_type, tp = 'all', condi = 'all')
    }
  }
  
  # Compute dispersion for each time point
  for (tp in 0:2) {
    filtered_df <- df %>% filter(Info_tpindex == tp)
    all_poa_df <- filtered_df %>%
      select(Dist_x, Dist_y, Dist_z) %>%
      drop_na()
    
    modalities <- filtered_df %>% select(modality)
    dist_matrix <- vegdist(all_poa_df, method = dist_metric)
    betadisper_results <- betadisper(dist_matrix, modalities$modality)
    permutest_results = permutest(betadisper_results, pairwise = TRUE, permutations = nb_permutations)
    add_results("all", tp, permutest_results, betadisper_results)
    
    # Plot if significant results
    if (permutest_results$tab$`Pr(>F)`[1] <= 0.05) {
      title <- sprintf("%s Dispersion Test: tp = %s, condi = %s, p_value = %.3f", analysis_type, tp, 'all', permutest_results$tab$`Pr(>F)`[1])
      plot(betadisper_results, main = title)
      
      if (!is.null(path_to_data)) {
        save_pca_plot(plot_func = function() plot(betadisper_results, main = title),
                      path_to_data = path_to_data, analysis_type = analysis_type, tp = tp, condi = 'all')
      }
    }
  }
  
  # Compute dispersion for each condition
  all_conditions <- unique(df$Info_condition)
  for (condi in all_conditions) {
    filtered_df <- df %>% filter(Info_condition == condi)
    all_poa_df <- filtered_df %>%
      select(Dist_x, Dist_y, Dist_z) %>%
      drop_na()
    
    modalities <- filtered_df %>% select(modality)
    dist_matrix <- vegdist(all_poa_df, method = dist_metric)
    betadisper_results <- betadisper(dist_matrix, modalities$modality)
    permutest_results = permutest(betadisper_results, pairwise = TRUE, permutations = nb_permutations)
    add_results(condi, "all", permutest_results, betadisper_results)
    
    # Plot if significant results
    if (permutest_results$tab$`Pr(>F)`[1] <= 0.05) {
      title <- sprintf("%s Dispersion Test: tp = %s, condi = %s, p_value = %.3f", analysis_type, 'all', condi, permutest_results$tab$`Pr(>F)`[1])
      plot(betadisper_results, main = title)
      
      if (!is.null(path_to_data)) {
        save_pca_plot(plot_func = function() plot(betadisper_results, main = title),
                      path_to_data = path_to_data, analysis_type = analysis_type, tp = 'all', condi = condi)
      }
    }
    
    # Compute dispersion for each time point within each condition
    for (tp in 0:2) {
      filtered_df_tp <- filtered_df %>% filter(Info_tpindex == tp)
      
      all_poa_df_tp <- filtered_df_tp %>%
        select(Dist_x, Dist_y, Dist_z) %>%
        drop_na()
      
      modalities_tp <- filtered_df_tp %>% select(modality)
      dist_matrix_tp <- vegdist(all_poa_df_tp, method = dist_metric)
      betadisper_results_tp <- betadisper(dist_matrix_tp, modalities_tp$modality)
      permutest_results_tp = permutest(betadisper_results_tp, pairwise = TRUE, permutations = nb_permutations)
      add_results(condi, tp, permutest_results_tp, betadisper_results_tp)
      
      # Plot if significant results
      if (permutest_results_tp$tab$`Pr(>F)`[1] <= 0.05) {
        title <- sprintf("%s Dispersion Test: tp = %s, condi = %s, p_value = %.3f", analysis_type, tp, condi, permutest_results_tp$tab$`Pr(>F)`[1])
        plot(betadisper_results_tp, main = title)
        
        if (!is.null(path_to_data)) {
          save_pca_plot(plot_func = function() plot(betadisper_results_tp, main = title),
                        path_to_data = path_to_data, analysis_type = analysis_type, tp = tp, condi = condi)
        }
      }
    }
  }
  
  return(results_dispersion_analysis)}


# Define a function to save PCA plots
save_pca_plot <- function(plot_func, path_to_data, analysis_type,  tp, condi, width = 800, height = 600, res = 100) {
  # Define directory path for saving the plot
  pca_plot_dir <- path(path_to_data, 'pca_plot_R')
  
  # Create directory if it does not exist
  dir_create(pca_plot_dir, recurse = TRUE)
  
  # Construct filename with parameters
  filename <- sprintf("all_subjects_analysis-%s_modality_comparison_analysis-dispersion_pca_plot_tp-%s_condi-%s_R.png", analysis_type, tp, condi)
  
  # Full path for saving the plot
  file_path <- file.path(pca_plot_dir, filename)
  
  # Ensure file_path is a character string and not empty
  file_path_char <- as.character(file_path)
  
  # Save the plot
  png(filename = file_path, width = width, height = height, res = res)
  plot_func()  # Call the function to create the plot
  dev.off()
}