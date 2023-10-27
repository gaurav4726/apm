start_time <- Sys.time()

library(arrow)
library(MatchIt)
library(grf)
library(magrittr)
library(mltools)
library(MLmetrics)
library(dplyr)
library(glue)
library(knitr)
library(reticulate)
library(parallel)

reticulate::install_miniconda()
setwd("/home/rstudio/mac")
print(getwd())

source("m3_helper_file.r")
source_python("config.py")


# load input data from data directory
max_days <- max(py$requirement_dict$Performace_days$Performace_Post_day)
file_name <- paste("/home/rstudio/mac/artifacts/regression_output_data_", max_days, "days.parquet", sep = "")
df_inputt <- arrow::read_parquet(file_name)
#df_inputt <- arrow::read_parquet("/home/rstudio/mac/artifacts/regression_output_data_30days.parquet")
#df_input <- distinct(df_input)
df_classificationn = arrow::read_parquet("/home/rstudio/mac/artifacts/classification_model_ouput.parquet")
#df_classification <- distinct(df_classification)

###Batch_Processing####
batch_size <- 7
all_treatment_ids <- unique(df_inputt$treatment_id)

batches <- list()
num_treatment_ids <- length(all_treatment_ids)

if (num_treatment_ids <= batch_size * 1.5) {
  batches[[1]] <- all_treatment_ids
} else {
  num_full_batches <- floor(num_treatment_ids / batch_size)
  for (i in 1:(num_full_batches + 1)) {
    batch_start <- (i - 1) * batch_size + 1
    batch_end <- min(i * batch_size, num_treatment_ids)
    batch <- all_treatment_ids[batch_start:batch_end]
    batches[[i]] <- batch
  }
  remaining_start <- num_full_batches * batch_size + 1
  remaining_ids <- all_treatment_ids[remaining_start:num_treatment_ids]
  
  if (length(remaining_ids) <= batch_size / 2) {
    batches[[num_full_batches]] <- c(batches[[num_full_batches]], remaining_ids)
  } else {
    num_full_batches <- num_full_batches + 1
    batches[[num_full_batches]] <- remaining_ids
  }
  batches <- batches[1:num_full_batches]
}

#### Create dataframes for each batch
for (iter_num in 1:length(batches)) {
  batch_name <- paste("df_input_", iter_num, sep = "")
  batch_ids <- batches[[iter_num]]
  batch_df <- df_inputt[df_inputt$treatment_id %in% batch_ids, ]
  assign(batch_name, batch_df)
  
  batch_name_cl <- paste("df_classification_", iter_num, sep = "")
  batch_ids_cl <- batches[[iter_num]]
  batch_df_cl <- df_classificationn[df_classificationn$treatment_id %in% batch_ids_cl, ]
  assign(batch_name_cl, batch_df_cl)
  
  df_input <- distinct(batch_df)
  df_classification <- distinct(batch_df_cl)
  
  # Get all the performance indicators to be measured as specified in the config file
  pm <- py$requirement_dict$input_dict$performance_matrix
  prediction_columns <- grep("predictions", names(df_input), value = TRUE)
  prediction_columns <- gsub(" predictions", "", prediction_columns)
  pm <- pm[(pm %in% prediction_columns)]
  print(pm)
  ###Batch_Processing####



  df_opp = merge(df_input,df_classification[c('emp_id','treatment_id','probability')],by = c("emp_id", "treatment_id"))
  names(df_opp)[names(df_opp) == 'probability'] <- 'weights'

  columns_to_remove <- grepl("predictions|__index_level_0__|treatment_label_taken", names(df_input))
  df_input <- df_input[, !columns_to_remove]

  # Initialize empty data frames for storing segmentwise and overall results
  segmentwise_output_main <- data.frame()
  overall_output_main <- data.frame()

#   # Get all the performance indicators to be measured as specified in the config file
#   pm <- py$requirement_dict$input_dict$performance_matrix
#   prediction_columns <- grep("predictions", names(df_input), value = TRUE)
#   prediction_columns <- gsub(" predictions", "", prediction_columns)
#   pm <- pm[!(pm %in% prediction_columns)]

  # Extract the base variable names without the "_pre" suffix
  base_names <- sub("_pre$", "", colnames(df_input))

  # Find the variable names that have both "pre" and "non-pre" versions
  post_data <- unique(base_names[duplicated(base_names)])

  for (outcome in pm) {
    # set up reserved column names, x_variables are not in the reserved set
    col_emp_id <- "emp_id"
    col_outcome <- outcome
    col_treatment <- "Control/Test_group"
    #treatment_id <- "treatment_id"
    treatment_id <- "treatment_id"
    post_data <- post_data
    colnames_reserved <- c(
      col_emp_id,
      col_outcome,
      col_treatment,
      treatment_id,
      post_data)
    x_vars <- setdiff(names(df_input), colnames_reserved)

    cols_to_keep <- grep(paste(outcome, "predictions", collapse = "|"), names(df_opp), value = TRUE, invert = TRUE)
    cols_to_keep <- c(cols_to_keep, paste(outcome, "predictions", sep = " "))
    df_opp <- df_opp[, cols_to_keep]

    # Match Strategy 1: Executing the first match. We implement the CEM method using the "ATT" estimand. It matches units 
    # based on a specified treatment variable and covariates. The cutpoints for the covariates are set to "q5", 
    # dividing them into five groups based on their values.
    match_result1 <- match_1(
      df = df_input,
      w_var = col_treatment,
      x_vars = x_vars)

    # Match Strategy 2: The second strategy implements CEM with multiple sets of cutpoints for matching. 
    # It takes the input data, treatment variable, covariates, and a list of cutpoints as parameters. 
    # The function iterates over each set of cutpoints and performs matching using the matchit function.

    # Defining the cutpoints for each covariate using "q5" as the cutpoint
    cutpoints2 <- lapply(x_vars, function(var) setNames(list("q3"), var))

    # Executing the 2nd match
    match_result2 <- match_2(df_input, col_treatment, x_vars, cutpoints2)

    # Match Strategy 3: We use the cut_log strategy with a constant of 0.5 for all numeric variables
    # and q5 for all non-numeric variables

    cutpoints3 <- list()

    # Apply cut_log strategy to each variable in x_vars
    for (var in x_vars) {
      # Check if the variable is numeric
      if (is.numeric(df_input[[var]])) {
        # Apply cut_log strategy using constant = 0.5
        cutpoints3[[var]] <- list(var = cut_log(df_input[[var]], constant = 0.6))
      } else {
        # For non-numeric variables, use "q5" as the cutpoint
        cutpoints3[[var]] <- list(var = "q5")
      }
    }

    # Executing the 3rd match
    match_result3 <- match_2(df_input, col_treatment, x_vars, cutpoints3)

    # Match Strategy 4: We use the cut_log strategy with a constant of 0.5 for only the first numeric
    # variable and q5 for all other variables

    cutpoints4 <- list()

    # Flag to check if the first numeric variable is found
    found_first_numeric <- FALSE

    # Apply cut_log strategy to the variables in x_vars
    for (var in x_vars) {
      if (is.numeric(df_input[[var]])) {
        if (!found_first_numeric) {
          # Apply cut_log strategy to the first numeric variable
          cutpoints4[[var]] <- list(var = cut_log(df_input[[var]], constant = 0.5))
          found_first_numeric <- TRUE
        } else {
          # For the remaining variables, use "q5" as the cutpoint
          cutpoints4[[var]] <- list(var = "q5")
        }
      } else {
        # For non-numeric variables, use "q5" as the cutpoint
        cutpoints4[[var]] <- list(var = "q5")
      }
    }

    # Executing the 4th match
    match_result4 <- match_2(df_input, col_treatment, x_vars, cutpoints4)

    # Match Strategy 5: We test different cutpoint strategies for matching by specifying different
    # cutpoint types (quantiles or widths) and values. It creates a list of cutpoints based on the covariates
    # and performs matching using the matchit function

    # Defining the cutpoint types and values to test
    cutpoint_types <- c("q", "w")
    cutpoint_values <- 2:5  # Test for 2, 3, 4, and 5 quantiles/widths

    # Executing the 5th match
    match_result5 <- match_3(df_input, col_treatment, x_vars, cutpoint_types, cutpoint_values)

    # Storing the results of each strategy in a table
    # Generate results table for match_result1
    results_table1 <- data.frame(
      Match_Num = 'match_result1',
      Cutpoint = 'q5',
      NumOriginal = nrow(df_input),
      NumMatched = sum(get_match_info(match_result1)$counts["Matched", ]),
      std_mean_diff_max_matched = get_match_info(match_result1)$std_mean_diff_max_matched
    )

    # Combine results tables for match_result2, match_result3, match_result4, match_result5
    combined_table <- results_table1

    match_results <- list(match_result2, match_result3, match_result4, match_result5)
    match_nums <- c("match_result2", "match_result3", "match_result4", "match_result5")

    # Iterate over each matching result and generate results table
    for (i in seq_along(match_results)) {
      results_table <- generate_results_table(match_results[[i]], match_nums[i])
      combined_table <- rbind(combined_table, results_table)
    }

    combined_table$std_mean_diff_max_matched <- as.numeric(combined_table$std_mean_diff_max_matched)

    # Add match categories based on std_mean_diff_max_matched values
    combined_table$Category <- ifelse(abs(combined_table$std_mean_diff_max_matched) < 0.05, "Green",
                                    ifelse(abs(combined_table$std_mean_diff_max_matched) < 0.10, "Yellow", "Red"))

    # Calculating match %
    combined_table$NumMatched <- as.numeric(combined_table$NumMatched)
    combined_table$NumOriginal <- as.numeric(combined_table$NumOriginal)
    combined_table$MatchPercent <- combined_table$NumMatched/combined_table$NumOriginal

    combined_table$std_mean_diff_max_matched <- as.numeric(combined_table$std_mean_diff_max_matched)

#     #Filtering out only those matches where the standardized mean difference < 0.05 and ranking by match percentage
#     combined_table <- combined_table %>% filter(Category=='Green')

#     # Initialize a log file
#     log_file_path <- "error_log.txt"
                         
#     # Create the log file if it doesn't exist
#     if (!file.exists(log_file_path)) {
#       file.create(log_file_path)
#     }

#     # Function to filter 'Green' category with tryCatch
#     filter_green_category <- function(combined_table, log_file_path) {
#       tryCatch({
#         # Load the dplyr package in the worker and export the %>%
#         library(dplyr, character.only = TRUE)
#         env <- new.env()
#         env$`%>%` <- dplyr::`%>%`

#         # Filtering out only those matches where the standardized mean difference < 0.05 and ranking by match percentage
#         combined_table <- combined_table %>% filter(Category == 'Green')
#         return(combined_table)
#       }, error = function(e) {
#         # Handle the exception by logging the error message to a file
#         error_message <- paste("An error occurred: ", conditionMessage(e))
#         cat(error_message, "\n", file = log_file_path, append = TRUE)
#         return(NULL) # Return NULL in case of an error
#       })
#     }

#     # Build a cluster object.
#     cl <- makeCluster(2)

#     # Export the necessary functions and objects to all nodes in the cluster
#     clusterExport(cl, c("filter_green_category", "log_file_path"))

#     # Make a list of inputs that will be handled concurrently.
#     inputs <- list(combined_table)

#     # Apply the filtering function in parallel to each entry of the list, passing the log file path
#     filtered_tables <- parLapply(cl, inputs, filter_green_category, log_file_path = log_file_path)

#     # Put an end to the cluster
#     stopCluster(cl)
               
    combined_table$Rank <- rank(-combined_table$MatchPercent)

    # Identifying the best match in terms of MatchPercent
    combined_table1 <- combined_table %>% filter(Rank == min(Rank))
    combined_table1 <- combined_table1[1, ]
    combined_table1$KPI <- outcome
    print(kable(combined_table1, format = "simple"))

    # Fetching final data for the best match identified
    for (match_obj_name in combined_table1$Cutpoint) {
      match_result_name <- gsub("'", "", combined_table1$Match_Num)

      if (match_result_name == "match_result1") {
      match_obj <- get(match_result_name)
    } else {
      match_obj <- get(match_result_name)[[match_obj_name]]
    }
      grf_ate_result <- grf_and_effect_estimate(match_obj)

      data <- grf_ate_result$data

      # Iterate over unique treatment_id values
      for (label in unique(data$treatment_id)) {
        subset <- data %>% filter(treatment_id == label)

        # Skip if subset is empty or has <5 rows
        if (nrow(subset) == 0 || nrow(subset) < 5) {
          next  # Skip to the next iteration of the "label" loop
        }

        tryCatch({
          grf_result_1 <- grf_fit(
            data_match = subset,
            x_vars = x_vars,
            y_var = col_outcome,
            w_var = col_treatment
          )

          grf_result_auc_1 = grf_result_1$W_auc
          print(paste0('AUC_grf_result:',grf_result_auc_1))

          subset_opp <- df_opp %>% filter(treatment_id == label)

          grf_result_opp <- grf_fit(
            data_match = subset_opp,
            x_vars = x_vars,
            y_var = col_outcome,
            w_var = col_treatment
          )

          grf_result_auc_2 = grf_result_opp$W_auc
          print(paste0('AUC_grf_result_opp:',grf_result_auc_2))

          if (grf_result_auc_1 > grf_result_auc_2) {
          grf_result <- grf_result_1
        } else {
          grf_result <- grf_result_opp
        }

          # Get treatment effect
          average_treatment_effect <- ate_fun(grf_result$cf)

          # Get the estimated ATE and standard error
          ate <- average_treatment_effect$estimate
          se <- average_treatment_effect$std_err

          alpha <- 0.05
          moe <- qnorm(1 - alpha/2) * se

          lower_bound <- ate - moe
          upper_bound <- ate + moe

          if (lower_bound <= 0 && upper_bound >= 0) {
            significance <- "Not Significant"
          } else {
            significance <- "Significant"
          }

          # Create a data frame with the results for the current treatment_id (for overall values)
          overall_result <- data.frame(
            KPI = outcome,
            treatment_id = label,
            Segment_Name = "Overall",
            Segment = "Overall",
            Average_treatment_effect = ate,
            Standard_Error = se,
            CI = paste0("[", round(lower_bound, 2), " - ", round(upper_bound, 2), "]"),
            Significance = significance
          )

          # Append the result to the overall output data frame
          overall_output_main <- rbind(overall_output_main, overall_result)
        }, error = function(e) {
          # Handle the error
          print(paste("Error occurred for treatment_id", label))
          print(e$message)
        })
      }
    }
    print("Overall Output")
    print(kable(overall_output_main, format = "simple"))


    for (j in seq_along(py$requirement_dict$customer_segemnts_for_output)) {
    segment_name <- paste0("segment", j)
    segments <- py$requirement_dict$customer_segemnts_for_output[[segment_name]]

    # Initialize an empty data frame to store the offering_id and segmentwise results
    segment_output <- data.frame()

    # Iterate over unique treatment_id values
    for (label in unique(data$treatment_id)) {
    subset <- data %>% filter(treatment_id == label)
    #unique_segments <- unique(subset[[segments]])
    for (seg in unique(data[[segments]])) {
      tryCatch({
        segment_subset <- subset %>% filter(!!sym(segments) == seg)

        # Skip if segment_subset is empty or has <5 rows
        if (nrow(segment_subset) == 0 || nrow(segment_subset) < 5) {
          next  # Skip to the next iteration of the "seg" loop
        }

        grf_result_1 <- grf_fit(
            data_match = subset,
            x_vars = x_vars,
            y_var = col_outcome,
            w_var = col_treatment
          )

          grf_result_auc_1 = grf_result_1$W_auc
          print(paste0('AUC_grf_result:',grf_result_auc_1))

          subset_opp <- df_opp %>% filter(treatment_id == label) %>% filter(!!sym(segments) == seg)

          grf_result_opp <- grf_fit(
            data_match = subset_opp,
            x_vars = x_vars,
            y_var = col_outcome,
            w_var = col_treatment
          )

          grf_result_auc_2 = grf_result_opp$W_auc
          print(paste0('AUC_grf_result_opp:',grf_result_auc_2))

          if (grf_result_auc_1 > grf_result_auc_2) {
          grf_result <- grf_result_1
        } else {
          grf_result <- grf_result_opp
        }

        # Generating the treatment effect
        average_treatment_effect <- ate_fun(grf_result$cf)

        # ATE estimate and standard error
        ate <- average_treatment_effect$estimate
        se <- average_treatment_effect$std_err

        alpha <- 0.05
        moe <- qnorm(1 - alpha/2) * se

        lower_bound <- ate - moe
        upper_bound <- ate + moe

        if (lower_bound <= 0 && upper_bound >= 0) {
          significance <- "Not Significant"
        } else {
          significance <- "Significant"
        }

        # Creating a data frame with the results for the current segment and treatment_id
        result_row <- data.frame(
          KPI = outcome,
          treatment_id = label,
          Segment_Name = segments,
          Segment = seg,
          Average_treatment_effect = ate,
          Standard_Error = se,
          CI = paste0("[", round(lower_bound, 2), " - ", round(upper_bound, 2), "]"),
          Significance = significance
        )

        # Appending the result to the final output data frame
        segment_output <- rbind(segment_output, result_row)
      }, error = function(e) {
        # Handle the error
        print(paste("Error occurred for segment", segment_name))
        print(e$message)
      })
    }

    # Skip to the next iteration if segment_output is empty
    if (nrow(segment_output) == 0) {
      next  # Skip to the next iteration of the "label" loop
    }

    # Subseting the data for the current treatment_id
    overall_subset <- data %>% filter(treatment_id == label)

    tryCatch({
      grf_result_1 <- grf_fit(
            data_match = subset,
            x_vars = x_vars,
            y_var = col_outcome,
            w_var = col_treatment
          )

          grf_result_auc_1 = grf_result_1$W_auc
          print(paste0('AUC_grf_result:',grf_result_auc_1))

          subset_opp <- df_opp %>% filter(treatment_id == label)

          grf_result_opp <- grf_fit(
            data_match = subset_opp,
            x_vars = x_vars,
            y_var = col_outcome,
            w_var = col_treatment
          )

          grf_result_auc_2 = grf_result_opp$W_auc
          print(paste0('AUC_grf_result_opp:',grf_result_auc_2))

          if (grf_result_auc_1 > grf_result_auc_2) {
          grf_result <- grf_result_1
        } else {
          grf_result <- grf_result_opp
        }

      # Get treatment effect
      average_treatment_effect <- ate_fun(grf_result$cf)

      # Get the estimated ATE and standard error
      ate <- average_treatment_effect$estimate
      se <- average_treatment_effect$std_err

      alpha <- 0.05
      moe <- qnorm(1 - alpha/2) * se

      lower_bound <- ate - moe
      upper_bound <- ate + moe

      if (lower_bound <= 0 && upper_bound >= 0) {
        significance <- "Not Significant"
      } else {
        significance <- "Significant"
      }

      # Create a data frame with the results for the current treatment_id (for overall values)
      overall_result <- data.frame(
        KPI = outcome,
        treatment_id = label,
        Segment_Name = segments,
        Segment = "Overall",
        Average_treatment_effect = ate,
        Standard_Error = se,
        CI = paste0("[", round(lower_bound, 2), " - ", round(upper_bound, 2), "]"),
        Significance = significance
      )

      #segment_output appended to the overall result
      segment_output <- rbind(segment_output, overall_result)
      segmentwise_output_main <- rbind(segmentwise_output_main,segment_output)
    }, error = function(e) {
      # Handle the error
      print(paste("Error occurred for overall segment"))
      print(e$message)
    })
  }
  }
    print("\n")
    print("Segmentwise Output")
    print(kable(segmentwise_output_main, format = "simple"))
  }

  # Dropping duplicates across the final tables
  overall_output_main = overall_output_main[!duplicated(overall_output_main), ]
  segmentwise_output_main = segmentwise_output_main[!duplicated(segmentwise_output_main), ]
  segmentwise_output_main = segmentwise_output_main %>% filter(Segment != 'Overall')

  # Saving the output as parquet files
  #write_parquet(overall_output_main, "/home/rstudio/mac/artifacts/m3_overall_output.parquet")
  #write_parquet(segmentwise_output_main, "/home/rstudio/mac/artifacts/m3_segmentwise_output.parquet")
  #write_parquet(combined_table,"/home/rstudio/mac/artifacts/m3_match_info.parquet")
  
  write_parquet(data,"/home/rstudio/mac/artifacts/match_data.parquet")
                   
  overall_out_name <- paste("overall_output_main_", iter_num, sep = "")
  assign(overall_out_name, overall_output_main)
  parquet_file_path <- paste("/home/rstudio/mac/artifacts/m3_overall_output_", iter_num, ".parquet", sep = "")
  write_parquet(overall_output_main, parquet_file_path)
  
  seg_out_name <- paste("segmentwise_output_main_", iter_num, sep = "")
  assign(seg_out_name, segmentwise_output_main)  
  parquet_file_path <- paste("/home/rstudio/mac/artifacts/m3_segmentwise_output_", iter_num, ".parquet", sep = "")
  write_parquet(segmentwise_output_main, parquet_file_path)
  
  comb_out_name <- paste("combined_table_", iter_num, sep = "")
  assign(comb_out_name, combined_table)
  parquet_file_path <- paste("/home/rstudio/mac/artifacts/m3_match_info_", iter_num, ".parquet", sep = "")
  write_parquet(combined_table,parquet_file_path)

  print("Output saved as Parquet file")
}

base_directory <- "/home/rstudio/mac/artifacts/"
file_name_pattern_seg <- "m3_segmentwise_output_%d.parquet"
combined_df_seg <- tibble::tibble()
file_name_pattern_over <- "m3_overall_output_%d.parquet"
combined_df_over <- tibble::tibble()
file_name_pattern_match_info <- "m3_match_info_%d.parquet"
combined_df_match_info <- tibble::tibble()

for (count in 1:length(batches)) {
  file_paths_seg <- sprintf(paste0(base_directory, file_name_pattern_seg), 1:count)
  file_paths_over <- sprintf(paste0(base_directory, file_name_pattern_over), 1:count)
  file_paths_match_info <- sprintf(paste0(base_directory, file_name_pattern_match_info), 1:count)
  dfs_seg <- lapply(file_paths_seg, function(file_path) {
    arrow::read_parquet(file_path) %>% as_tibble()
  })
  dfs_over <- lapply(file_paths_over, function(file_path) {
    arrow::read_parquet(file_path) %>% as_tibble()
  })
  dfs_match_info <- lapply(file_paths_match_info, function(file_path) {
    arrow::read_parquet(file_path) %>% as_tibble()
  })
  combined_df_seg <- dplyr::bind_rows(dfs_seg)
  combined_df_over <- dplyr::bind_rows(dfs_over)
  combined_df_match_info <- dplyr::bind_rows(dfs_match_info)
}

output_file_path_seg <- file.path(base_directory, "m3_segmentwise_output.parquet")
arrow::write_parquet(combined_df_seg, output_file_path_seg)
output_file_path_over <- file.path(base_directory, "m3_overall_output.parquet")
arrow::write_parquet(combined_df_over, output_file_path_over)
output_file_path_match_info <- file.path(base_directory, "m3_match_info.parquet")
arrow::write_parquet(combined_df_match_info, output_file_path_match_info)

end_time <- Sys.time()
time_diff <- as.numeric(difftime(end_time, start_time, units = "mins"))
cat("Run time:", time_diff, "minutes\n")
