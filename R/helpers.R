l0learn_matrix <- function(task, feature_names) {
  x <- as.matrix(task$data(cols = feature_names))
  storage.mode(x) <- "double"
  x
}

l0learn_n_folds <- function(pv) {
  if (is.null(pv$nFolds)) 10L else pv$nFolds
}

l0learn_check_grid <- function(x) {
  if (!is.list(x)) {
    return("Must be a list of numeric vectors")
  }
  for (grid in x) {
    valid <- is.numeric(grid) && length(grid) >= 1L && !anyNA(grid) &&
      all(grid > 0) && all(diff(grid) < 0)
    if (!valid) {
      return("Each element must be a strictly decreasing vector of positive numbers")
    }
  }
  TRUE
}

l0learn_prepare_args <- function(pv, n_features) {
  has_bounds <- (!is.null(pv$lows) && is.finite(pv$lows)) ||
    (!is.null(pv$highs) && is.finite(pv$highs))
  if (has_bounds && identical(pv$algorithm, "CDPSI")) {
    mlr3misc::stopf("Bounds (lows, highs) are not supported by algorithm = \"CDPSI\".")
  }
  if (!is.null(pv$lows) && !is.null(pv$highs) && pv$lows >= pv$highs) {
    mlr3misc::stopf("lows must be smaller than highs.")
  }
  if (!is.null(pv$excludeFirstK) && pv$excludeFirstK > n_features) {
    mlr3misc::stopf(
      "excludeFirstK = %i is larger than the number of features (%i).",
      pv$excludeFirstK, n_features)
  }
  penalty <- if (is.null(pv$penalty)) "L0" else pv$penalty
  n_grids <- length(pv$lambdaGrid)
  if (penalty == "L0" && n_grids > 1L) {
    mlr3misc::stopf("penalty = \"L0\" requires lambdaGrid to contain exactly one vector.")
  }
  if (penalty != "L0" && n_grids > 0L) {
    pv$nGamma <- n_grids
  }
  pv
}

l0learn_cvfit <- function(x, y, pv) {
  seed_name <- ".Random.seed"
  global_env <- globalenv()
  had_seed <- exists(seed_name, envir = global_env, inherits = FALSE)
  if (had_seed) {
    old_seed <- get(seed_name, envir = global_env, inherits = FALSE)
  }
  on.exit({
    if (had_seed) {
      assign(seed_name, old_seed, envir = global_env)
    } else if (exists(seed_name, envir = global_env, inherits = FALSE)) {
      rm(list = seed_name, envir = global_env)
    }
  }, add = TRUE)
  tryCatch(
    mlr3misc::invoke(
      L0Learn::L0Learn.cvfit,
      x = x, y = y, loss = "SquaredError",
      .args = pv),
    error = function(e) {
      message_text <- conditionMessage(e)
      if (grepl("out of bounds|out_of_range", message_text)) {
        mlr3misc::stopf(
          "L0Learn::L0Learn.cvfit() failed for %i rows and nFolds = %i: %s. Its fold splitting breaks for some combinations of rows and folds, so choose another value of nFolds.",
          nrow(x), l0learn_n_folds(pv), message_text)
      }
      mlr3misc::stopf(
        "L0Learn::L0Learn.cvfit() failed: %s", message_text)
    })
}

l0learn_select <- function(fit, selection, n_folds, n_rows) {
  cv_means <- lapply(fit$cvMeans, as.numeric)
  cv_sds <- lapply(fit$cvSDs, as.numeric)
  if (all(is.na(unlist(cv_means)))) {
    mlr3misc::stopf(
      "L0Learn::L0Learn.cvfit() returned only NaN cross-validation errors for %i rows and nFolds = %i. Choose another value of nFolds.",
      n_rows, n_folds)
  }
  best_cv <- vapply(cv_means, function(cv) {
    if (all(is.na(cv))) Inf else min(cv, na.rm = TRUE)
  }, numeric(1))
  g <- which.min(best_cv)
  l <- max(which(cv_means[[g]] == best_cv[g]))
  if (selection == "1se") {
    if (is.na(cv_sds[[g]][l])) {
      mlr3misc::warningf(
        "The standard error of the cross-validated error is not available, so the minimum is used instead of the 1se rule.")
    } else {
      threshold <- best_cv[g] + cv_sds[[g]][l] / sqrt(n_folds)
      candidates <- which(cv_means[[g]] <= threshold)
      l <- candidates[which.min(fit$fit$suppSize[[g]][candidates])]
    }
  }
  list(
    fit = fit,
    gamma_index = g,
    lambda_index = l,
    lambda = fit$fit$lambda[[g]][l],
    gamma = fit$fit$gamma[g],
    support_size = as.integer(fit$fit$suppSize[[g]][l]),
    cv_mean = cv_means[[g]][l],
    cv_sd = cv_sds[[g]][l])
}

l0learn_coef <- function(model) {
  path <- model$fit$fit
  g <- model$gamma_index
  l <- model$lambda_index
  list(
    a0 = path$a0[[g]][l],
    beta = as.numeric(path$beta[[g]][, l]))
}
