simulate_sparse_task <- function(n = 200L, p = 50L, n_true = 5L, seed = 1L) {
  set.seed(seed)
  x <- matrix(
    stats::rnorm(n * p), n, p,
    dimnames = list(NULL, paste0("x", seq_len(p))))
  beta <- c(rep(2, n_true), rep(0, p - n_true))
  y <- as.numeric(x %*% beta + stats::rnorm(n))
  mlr3::as_task_regr(data.frame(y = y, x), target = "y", id = "sparse")
}

test_that("learner is registered with the expected interface", {
  learner <- mlr3::lrn("regr.L0Learn")
  expect_s3_class(learner, "LearnerRegrL0Learn")
  expect_identical(learner$id, "regr.L0Learn")
  expect_identical(learner$predict_types, "response")
  expect_identical(learner$packages, c("mlr3", "L0Learn"))
  expect_true("selected_features" %in% learner$properties)
  tags <- learner$param_set$tags
  expect_true(all(vapply(tags, function(tag) "train" %in% tag, logical(1))))
})

test_that("param_set validates values and dependencies", {
  learner <- mlr3::lrn("regr.L0Learn")
  expect_error(learner$param_set$set_values(penalty = "L2"))
  expect_error(learner$param_set$set_values(nFolds = 1L))
  expect_error(learner$param_set$set_values(nGamma = 3L))
  learner$param_set$set_values(penalty = "L0L2", nGamma = 3L)
  expect_identical(learner$param_set$values$nGamma, 3L)
})

test_that("train and predict work on mtcars", {
  task <- mlr3::tsk("mtcars")
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L, maxSuppSize = 5L)
  learner$train(task)
  prediction <- learner$predict(task)
  expect_s3_class(prediction, "PredictionRegr")
  expect_type(prediction$response, "double")
  expect_length(prediction$response, task$nrow)
  expect_false(anyNA(prediction$response))
  expect_true(all(is.finite(prediction$response)))
})

test_that("predict works on new rows and reordered columns", {
  task <- mlr3::tsk("mtcars")
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L)
  learner$train(task, row_ids = 1:24)
  prediction <- learner$predict(task, row_ids = 25:32)
  expect_length(prediction$response, 8L)
  new_data <- data.table::as.data.table(task$data(rows = 25:32))
  reordered <- new_data[, rev(names(new_data)), with = FALSE]
  expect_equal(learner$predict_newdata(reordered)$response, prediction$response)
})

test_that("predictions equal intercept plus linear combination", {
  task <- simulate_sparse_task()
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L)
  learner$train(task)
  path <- learner$model$fit$fit
  g <- learner$model$gamma_index
  l <- learner$model$lambda_index
  x <- as.matrix(task$data(cols = learner$state$feature_names))
  expected <- path$a0[[g]][l] + as.numeric(x %*% path$beta[[g]][, l])
  expect_equal(learner$predict(task)$response, expected)
})

test_that("learner beats featureless on sparse simulated data", {
  task <- simulate_sparse_task()
  split <- mlr3::partition(task, ratio = 0.7)
  mse <- function(learner) {
    learner$train(task, row_ids = split$train)
    learner$predict(task, row_ids = split$test)$score(mlr3::msr("regr.mse"))
  }
  mse_l0 <- mse(mlr3::lrn("regr.L0Learn", nFolds = 5L))
  mse_featureless <- mse(mlr3::lrn("regr.featureless"))
  expect_lt(mse_l0, mse_featureless / 5)
})

test_that("selected features recover the true support", {
  task <- simulate_sparse_task()
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L)
  learner$train(task)
  selected <- learner$selected_features()
  expect_type(selected, "character")
  expect_true(all(paste0("x", 1:5) %in% selected))
  expect_lt(length(selected), 15L)
})

test_that("selected features are correct without an intercept", {
  set.seed(1)
  x <- matrix(rnorm(1000), 100, dimnames = list(NULL, paste0("x", 1:10)))
  y <- as.numeric(x[, 1:3] %*% c(3, 2, 1) + rnorm(100))
  task <- mlr3::as_task_regr(data.frame(y = y, x), target = "y")
  learner <- mlr3::lrn("regr.L0Learn", intercept = FALSE, nFolds = 3L)
  learner$train(task)
  expect_true(all(c("x1", "x2", "x3") %in% learner$selected_features()))
  expect_true(all(learner$selected_features() %in% colnames(x)))
  expect_false(anyNA(learner$predict(task)$response))
})

test_that("tied lambdas on the path still give one prediction per row", {
  set.seed(94)
  data <- data.frame(int = sample(1:3, 30, TRUE), target = rnorm(30))
  task <- mlr3::as_task_regr(data, target = "target")
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 3L)
  learner$train(task)
  expect_length(learner$predict_newdata(data)$response, 30L)
})

test_that("ties in the CV error do not select the empty model", {
  set.seed(100)
  y <- seq(-10, 10, length.out = 100)
  data <- data.frame(
    y = y, x = y + stats::rnorm(100, mean = 1), unimportant = stats::runif(100))
  task <- mlr3::as_task_regr(data, target = "y")
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 3L)
  learner$train(task)
  expect_true("x" %in% learner$selected_features())
  mse <- learner$predict(task)$score(mlr3::msr("regr.mse"))
  expect_lt(mse, 5)
})

test_that("L0L2 penalty selects gamma from the grid", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  learner <- mlr3::lrn(
    "regr.L0Learn", penalty = "L0L2", nGamma = 3L, nLambda = 20L, nFolds = 3L)
  learner$train(task)
  expect_length(learner$model$fit$fit$gamma, 3L)
  expect_identical(
    learner$model$gamma, learner$model$fit$fit$gamma[learner$model$gamma_index])
  expect_length(learner$predict(task)$response, task$nrow)
})

test_that("the seed controls the cross-validation folds", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  cv_means <- function(seed) {
    learner <- mlr3::lrn("regr.L0Learn", nFolds = 3L, seed = seed)
    learner$train(task)
    learner$model$fit$cvMeans
  }
  expect_identical(cv_means(7L), cv_means(7L))
  expect_false(identical(cv_means(7L), cv_means(8L)))
})

test_that("training does not change the global random number stream", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  set.seed(42)
  expected <- stats::runif(3)
  set.seed(42)
  mlr3::lrn("regr.L0Learn", nFolds = 3L)$train(task)
  expect_identical(stats::runif(3), expected)
})

test_that("unsupported fold counts give an informative error", {
  expect_error(
    mlr3::lrn("regr.L0Learn", nFolds = 10L)$train(mlr3::tsk("mtcars")),
    "nFolds = 10")
  set.seed(1)
  data <- data.frame(y = rnorm(45), a = rnorm(45), b = rnorm(45))
  task <- mlr3::as_task_regr(data, target = "y")
  expect_error(
    mlr3::lrn("regr.L0Learn", nFolds = 10L)$train(task),
    "NaN cross-validation errors")
})

test_that("selected_features() errors before training", {
  expect_error(
    mlr3::lrn("regr.L0Learn")$selected_features(), "not been trained")
})

test_that("resample and a cloned learner work", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 3L)
  result <- mlr3::resample(task, learner, mlr3::rsmp("cv", folds = 3L))
  scores <- result$score(mlr3::msr("regr.mse"))
  expect_identical(nrow(scores), 3L)
  expect_false(anyNA(scores$regr.mse))
  cloned <- learner$clone(deep = TRUE)
  cloned$param_set$set_values(nFolds = 4L)
  expect_identical(learner$param_set$values$nFolds, 3L)
})

test_that("the model stores the selected support and CV summary", {
  task <- simulate_sparse_task()
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L)
  learner$train(task)
  model <- learner$model
  expect_identical(model$support_size, length(learner$selected_features()))
  expect_true(is.finite(model$cv_mean))
  expect_true(is.finite(model$cv_sd))
  expect_equal(
    model$cv_mean, min(unlist(model$fit$cvMeans), na.rm = TRUE))
})

test_that("the 1se rule gives a model that is no larger and within one SE", {
  task <- simulate_sparse_task()
  fit_with <- function(selection) {
    learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L, selection = selection)
    learner$train(task)
    learner$model
  }
  min_model <- fit_with("min")
  se_model <- fit_with("1se")
  expect_lte(se_model$support_size, min_model$support_size)
  expect_lte(se_model$cv_mean, min_model$cv_mean + min_model$cv_sd / sqrt(5))
  expect_lte(se_model$lambda_index, min_model$lambda_index)
})

test_that("the 1se rule is strictly sparser on noisy data", {
  set.seed(1)
  n <- 150L
  p <- 20L
  x <- matrix(
    stats::rnorm(n * p), n, p, dimnames = list(NULL, paste0("x", seq_len(p))))
  y <- as.numeric(x[, 1:8] %*% rep(1, 8) + stats::rnorm(n, sd = 4))
  task <- mlr3::as_task_regr(data.frame(y = y, x), target = "y")
  sizes <- vapply(c("min", "1se"), function(selection) {
    learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L, selection = selection)
    learner$train(task)
    learner$model$support_size
  }, integer(1))
  expect_lt(sizes[["1se"]], sizes[["min"]])
})

test_that("selection = min keeps the default behaviour", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  default <- mlr3::lrn("regr.L0Learn", nFolds = 3L)$train(task)
  explicit <- mlr3::lrn("regr.L0Learn", nFolds = 3L, selection = "min")
  explicit$train(task)
  expect_equal(
    default$predict(task)$response, explicit$predict(task)$response)
})

test_that("bounds restrict the coefficients", {
  task <- simulate_sparse_task()
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L, lows = -0.5, highs = 0.5)
  learner$train(task)
  beta <- as.numeric(l0learn_coef(learner$model)$beta)
  expect_true(all(beta >= -0.5 & beta <= 0.5))
  expect_true(any(beta != 0))
})

test_that("invalid bound settings are rejected", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  expect_error(
    mlr3::lrn("regr.L0Learn", algorithm = "CDPSI", lows = -1)$train(task),
    "CDPSI")
  expect_error(
    mlr3::lrn("regr.L0Learn", lows = 0, highs = 0)$train(task),
    "smaller than highs")
  expect_error(mlr3::lrn("regr.L0Learn", lows = 1))
  expect_error(mlr3::lrn("regr.L0Learn", highs = -1))
})

test_that("a custom lambda grid is used", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  learner <- mlr3::lrn(
    "regr.L0Learn", nFolds = 3L, lambdaGrid = list(c(1, 0.1, 0.01)))
  learner$train(task)
  expect_equal(learner$model$fit$fit$lambda[[1]], c(1, 0.1, 0.01))
  expect_true(learner$model$lambda %in% c(1, 0.1, 0.01))
})

test_that("invalid lambda grids are rejected", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  expect_error(mlr3::lrn("regr.L0Learn", lambdaGrid = list(c(1, 2))))
  expect_error(mlr3::lrn("regr.L0Learn", lambdaGrid = list(c(1, -1))))
  expect_error(mlr3::lrn("regr.L0Learn", lambdaGrid = c(1, 0.1)))
  expect_error(
    mlr3::lrn(
      "regr.L0Learn", lambdaGrid = list(c(1, 0.1), c(1, 0.1)))$train(task),
    "exactly one vector")
})

test_that("a grid per gamma works with the L0L2 penalty", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  learner <- mlr3::lrn(
    "regr.L0Learn", penalty = "L0L2", nFolds = 3L,
    lambdaGrid = list(c(1, 0.1), c(1, 0.1, 0.01)))
  learner$train(task)
  expect_length(learner$model$fit$fit$gamma, 2L)
  expect_length(learner$predict(task)$response, task$nrow)
})

test_that("excludeFirstK larger than the number of features is refused", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 3L, excludeFirstK = 21L)
  expect_error(learner$train(task), "excludeFirstK = 21")
  learner$param_set$set_values(excludeFirstK = 2L)
  learner$train(task)
  expect_true(learner$model$support_size >= 2L)
})

test_that("other L0Learn errors are not blamed on nFolds", {
  set.seed(1)
  x <- matrix(stats::rnorm(200), 50, 4)
  y <- stats::rnorm(50)
  error <- tryCatch(
    l0learn_cvfit(x, y, list(nFolds = 3L, lows = 1)), error = conditionMessage)
  expect_match(error, "Bounds must conform")
  expect_false(grepl("fold splitting", error))
})

make_cv_fit <- function(means, sds, supp) {
  list(
    fit = list(
      lambda = list(seq_along(means)),
      gamma = 0,
      suppSize = list(supp)),
    cvMeans = list(means),
    cvSDs = list(sds))
}

test_that("1se picks the smallest support among the candidates", {
  fit <- make_cv_fit(c(5, 2.05, 2.04, 2), rep(1, 4), c(0, 4, 1, 5))
  chosen <- l0learn_select(fit, "1se", 4L, 100L)
  expect_identical(chosen$lambda_index, 3L)
  expect_identical(chosen$support_size, 1L)
  minimum <- l0learn_select(fit, "min", 4L, 100L)
  expect_identical(minimum$lambda_index, 4L)
})

test_that("1se falls back to the minimum with a warning without a SD", {
  fit <- make_cv_fit(c(5, 2.05, 2), c(NaN, NaN, NaN), c(0, 4, 5))
  expect_warning(
    chosen <- l0learn_select(fit, "1se", 4L, 100L), "standard error")
  expect_identical(chosen$lambda_index, 3L)
})

test_that("a gamma with only NaN errors is ignored without warnings", {
  fit <- list(
    fit = list(
      lambda = list(1:3, 1:3),
      gamma = c(1, 2),
      suppSize = list(c(0, 1, 2), c(0, 1, 2))),
    cvMeans = list(c(NaN, NaN, NaN), c(3, 2, 2.5)),
    cvSDs = list(c(NaN, NaN, NaN), c(1, 1, 1)))
  expect_no_warning(chosen <- l0learn_select(fit, "min", 5L, 100L))
  expect_identical(chosen$gamma_index, 2L)
  expect_identical(chosen$lambda_index, 2L)
})

test_that("1se works with several gammas", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  learner <- mlr3::lrn(
    "regr.L0Learn", penalty = "L0L2", nGamma = 3L, nLambda = 20L,
    nFolds = 3L, selection = "1se")
  learner$train(task)
  expect_length(learner$predict(task)$response, task$nrow)
  expect_true(learner$model$gamma_index %in% 1:3)
})

test_that("default bound values are accepted with CDPSI", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  learner <- mlr3::lrn(
    "regr.L0Learn", algorithm = "CDPSI", lows = -Inf, highs = Inf,
    nFolds = 3L)
  expect_no_error(learner$train(task))
})

test_that("a one-sided bound works", {
  task <- simulate_sparse_task()
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L, lows = 0)
  learner$train(task)
  expect_true(all(l0learn_coef(learner$model)$beta >= 0))
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L, highs = 0)
  learner$train(task)
  expect_true(all(l0learn_coef(learner$model)$beta <= 0))
})

test_that("a grid with penalty L0L2 does not warn about nGamma", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  learner <- mlr3::lrn(
    "regr.L0Learn", penalty = "L0L2", nFolds = 3L,
    lambdaGrid = list(c(1, 0.1), c(1, 0.1, 0.01)))
  expect_no_warning(learner$train(task))
})
