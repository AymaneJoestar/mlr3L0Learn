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
