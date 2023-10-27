# All packages required to run the file
packages <- c("MLmetrics", "arrow", "MatchIt","grf","magrittr","mltools","reticulate","parallel")

# Check if the above packages are already installed, if not install them
for (package in packages) {
  # Check if the package is already installed
  if (!require(package, character.only = TRUE)) {
    # If not installed, install the package
    install.packages(package)
  }
}
# Check if Miniconda is installed
if (!file.exists(reticulate::conda_binary())) {
  reticulate::install_miniconda()
}

# Match Strategy 1: We implement the CEM method using the "ATT" estimand. It matches units based on 
# a specified treatment variable and covariates. The cutpoints for the covariates are set to "q5", 
# dividing them into five groups based on their values.
match_1 <- function(df, w_var, x_vars) {
  w_var <- paste0("`", w_var, "`")
  x_vars <- paste0("`", x_vars, "`")

  MatchIt::matchit(
    formula = as.formula(paste(w_var, "~", paste(x_vars, collapse = " + "))),
    data = df %>% dplyr::mutate_if(is.character, as.factor),
    method = "cem",
    estimand = "ATT",
    k2k = FALSE,
    cutpoints = "q5"
  )
}

# match info function
get_match_info <- function(match_result) {
  match_summary <- summary(match_result)
  matchinfo <- list(
    compare_all     = match_summary$sum.all     %>% as.data.frame(),
    compare_matched = match_summary$sum.matched %>% as.data.frame(),
    counts          = match_summary$nn          %>% as.data.frame()
  )
  # get dataframe with single column containing the standardized mean diff
  df <- matchinfo$compare_matched["Std. Mean Diff."]
  idx <- df[[1]] %>% abs() %>% which.max()
  matchinfo$std_mean_diff_max_matched      <- df[[idx, 1]]
  matchinfo$std_mean_diff_max_matched_name <- rownames(df)[idx]
  return(matchinfo)
}

# remove columns with only 1 unique value
# NA values are not counted
remove_constant_columns <- function(df) {
  return(Filter(function(x) length(unique(x[!is.na(x)])) > 1, df))
}

get_x_matrix <- function(data_in, x_vars) {
  # prepare data, omit dependent variable, treatment indicator, propensity scores
  # remove constant columns
  data_model <- data_in %>%
    dplyr::select(
      tidyselect::all_of(x_vars)
    ) %>%
    remove_constant_columns()

  # causal_forest requires all numeric inputs, one_hot converts categorical
  # with one-hot encoding
  Xmatrix <- mltools::one_hot(
    data_model %>%
      dplyr::mutate_if(is.character, as.factor) %>%
      data.table::as.data.table(),
    dropUnusedLevels = TRUE) %>%
    as.matrix()
  return(Xmatrix)
}

# fit generalized random forest, estimate treatment effect
grf_fit <- function(data_match, x_vars, y_var, w_var) {
  Xmatrix <- get_x_matrix(data_match, x_vars)
  num.trees <- 2000
  honesty <- TRUE
  tune.parameters <- "none"

  # Convert Y, W, and other relevant variables to vectors
  Y <- as.vector(data_match[[y_var]])
  W <- as.vector(data_match[[w_var]])

  # fit causal forest
  cf_result <- grf::causal_forest(
    X = Xmatrix,
    Y = Y,
    W = W,
    sample.weights = data_match$weights,
    clusters = as.factor(data_match$subclass),
    tune.parameters = tune.parameters,
    num.trees = num.trees,
    honesty = honesty
  )
  # R squared is correlation squared
  Y_r2 <- cor(cf_result$Y.hat, cf_result$Y.orig)^2

  # assemble results
  return(list(
    cf = cf_result,
    W_auc = MLmetrics::AUC(cf_result$W.hat, cf_result$W.orig),
    Y_r2 = Y_r2,
    Y_cor = sqrt(Y_r2),
    Y_orig_mean = mean(cf_result$Y.orig),
    y_var = y_var,
    w_var = w_var,
    Xmatrix = Xmatrix
  ))
}


# get average treatment effect
ate_fun <- function(cf_result) {
  ate_result <- grf::average_treatment_effect(
    forest = cf_result,
    target.sample = 'overlap',
    method = 'AIPW'
  )
  
  # Extract the estimate and standard error separately
  estimate <- ate_result[1]
  std_err <- ate_result[2]
  
  # Return the estimate and standard error as a list
  return(list(estimate = estimate, std_err = std_err))
}

# run grf and get average treatment effect
grf_and_effect_estimate <- function(match_obj) {
  # get match data with groups
  data_match <- MatchIt::match.data(match_obj)

  # fit generalized random forest model
  grf_result <- grf_fit(
    data_match = data_match,
    x_vars = x_vars,
    y_var = col_outcome,
    w_var = col_treatment)

  # get treatment effect
  average_treatment_effect <- ate_fun(grf_result$cf)

  # collect results
  result <- list(
    data_match = data_match,
    grf_result = grf_result,
    average_treatment_effect = average_treatment_effect
  )
  return(result)
}


# log transforms data then cuts into equal width bins
# constant is the linear multiplier bin increment
cut_log <- function(x, constant = 0.3, min_width = 1000) {
  # adjust to avoid log(0) if necessary
  adjust <- any(x == 0)
  tol <- 1
  if (adjust) {
    xa <- x - min(x) + tol
  } else {
    xa <- x
  }
  xa_range <- log(xa) %>% range()
  step <- log(constant + 1)
  logcuts <- seq(from = xa_range[1], to = xa_range[2] + step, by = step)
  if (adjust) {
    cuts <- exp(logcuts) + min(x) - tol
  } else {
    cuts <- exp(logcuts)
  }
  # bins smaller than min_width are omitted
  idx <- cuts >= min_width
  # keep the first bin
  idx[1] <- TRUE
  cuts[idx]
}

# Match Strategy 2: The second strategy implements CEM with multiple sets of cutpoints for matching. 
# It takes the input data, treatment variable, covariates, and a list of cutpoints as parameters. 
# The function iterates over each set of cutpoints and performs matching using the matchit function.

match_2 <- function(df, w_var, x_vars, cutpoints) {
  formula_str <- paste(w_var, "~", paste("`", x_vars, "`", sep = "", collapse = " + "))
  w_var <- paste0("`", w_var, "`")
  x_vars <- paste0("`", x_vars, "`")
  formula <- as.formula(formula_str)

  # Initializing a list to store the match results
  match_results <- list()

  # Matching for each set of cutpoints
  for (i in seq_along(cutpoints)) {
    # Performing the matching
    match_result <- matchit(formula = as.formula(paste(w_var, "~", paste(x_vars, collapse = " + "))),
                            data = df %>% dplyr::mutate_if(is.character, as.factor),
                            method = "cem",
                            cutpoints = cutpoints[[i]])

    # Storing the match result
    match_results[[paste(x_vars[i], names(cutpoints[[i]]), sep = "_")]] <- match_result
  }

  return(match_results)
}

# Match Strategy 5: We test different cutpoint strategies for matching by specifying different
# cutpoint types (quantiles or widths) and values. It creates a list of cutpoints based on the covariates
# and performs matching using the matchit function

match_3 <- function(df, w_var, x_vars, cutpoint_types, cutpoint_values) {
  formula_str <- paste(w_var, "~", paste("`", x_vars, "`", sep = "", collapse = " + "))
  w_var <- paste0("`", w_var, "`")
  x_vars <- paste0("`", x_vars, "`")
  formula <- as.formula(formula_str)

  # Initializing a list to store the match results
  match_results <- list()

  # Performing the matching for each cutpoint type and value
  for (cutpoint_type in cutpoint_types) {
    for (cutpoint_value in cutpoint_values) {
      # Creating the cutpoints list
      cutpoints <- lapply(x_vars, function(x) paste0(cutpoint_type, cutpoint_value))
      names(cutpoints) <- x_vars

      # Perform the matching
      match_result <- matchit(formula = as.formula(paste(w_var, "~", paste(x_vars, collapse = " + "))),
                              data = df %>% dplyr::mutate_if(is.character, as.factor),
                              method = "cem",
                              cutpoints = cutpoints)

      match_results[[paste(cutpoint_type, cutpoint_value, sep = "_")]] <- match_result
    }
  }

  return(match_results)
}

# Function to generate results table for a specific matching strategy
generate_results_table <- function(match_result, match_num) {
  results_list <- vector("list", length(match_result))  # Preallocate list
  
  # Iterate over each matchit object in the match_result list
  for (i in seq_along(match_result)) {
    match_obj <- match_result[[i]]
    
    # Extract matched data
    matched_data <- MatchIt::match.data(match_obj)
    
    # Calculate the number of original and matched observations
    num_original <- nrow(df_input)
    num_matched <- sum(get_match_info(match_obj)$counts["Matched (ESS)", c("Control", "Treated")])
    
    # Obtain std_mean_diff_max_matched
    match_info <- get_match_info(match_obj)$std_mean_diff_max_matched
    
    # Store results in the preallocated list
    results_list[[i]] <- list(
      Match_Num = match_num,
      Cutpoint = names(match_result)[i],
      NumOriginal = num_original,
      NumMatched = num_matched,
      std_mean_diff_max_matched = match_info
    )
  }
  
  # Convert the results list to a data frame
  results_table <- do.call(rbind, results_list)
  
  return(results_table)
}
