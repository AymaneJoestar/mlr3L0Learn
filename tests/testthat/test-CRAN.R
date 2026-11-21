simulate_sparse_task <- function(n = 200L, p = 50L, n_true = 5L, seed = 1L) {
  set.seed(seed)
  x <- matrix(stats::rnorm(n * p), n, p, dimnames = list(NULL, paste0("x", seq_len(p))))
  beta <- c(rep(2, n_true), rep(0, p - n_true))
  y <- as.numeric(x %*% beta + stats::rnorm(n))
  data <- data.frame(y = y, x)
  mlr3::as_task_regr(data, target = "y", id = "sparse")
}

test_that("learner is registered and valid", {
  learner <- mlr3::lrn("regr.L0Learn")
  expect_s3_class(learner, "LearnerRegrL0Learn")
  expect_identical(learner$id, "regr.L0Learn")
  expect_identical(learner$predict_types, "response")
  expect_true("selected_features" %in% learner$properties)
  expect_true(all(c("train") %in% unlist(learner$param_set$tags)))
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

test_that("predict works on new rows with integer features", {
  task <- mlr3::tsk("mtcars")
  learner <- mlr3::lrn("regr.L0Learn", nFolds = 5L)
  learner$train(task, row_ids = 1:24)
  prediction <- learner$predict(task, row_ids = 25:32)
  expect_length(prediction$response, 8L)
  expect_false(anyNA(prediction$response))
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

test_that("L0L2 penalty selects gamma on a grid", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  learner <- mlr3::lrn(
    "regr.L0Learn", penalty = "L0L2", nGamma = 3L, nLambda = 20L, nFolds = 3L)
  learner$train(task)
  expect_true(learner$model$gamma %in% learner$model$fit$fit$gamma)
  expect_length(learner$predict(task)$response, task$nrow)
})

test_that("the seed makes training reproducible", {
  task <- simulate_sparse_task(n = 100L, p = 20L)
  fit_once <- function() {
    learner <- mlr3::lrn("regr.L0Learn", nFolds = 3L, seed = 7L)
    learner$train(task)
    learner$predict(task)$response
  }
  expect_equal(fit_once(), fit_once())
})

test_that("selected_features() errors before training", {
  expect_error(mlr3::lrn("regr.L0Learn")$selected_features(), "not been trained")
})

test_that("the learner passes mlr3 checks", {
  skip_if_not_installed("mlr3")
  learner <- mlr3::lrn("regr.L0Learn")
  expect_true(inherits(learner, "Learner"))
  expect_true(learner$param_set$check(list(penalty = "L0L1", nFolds = 5L)))
  expect_error(learner$param_set$values <- list(penalty = "L2"))
})
