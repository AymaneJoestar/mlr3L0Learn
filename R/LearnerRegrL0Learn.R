#' @title Regression Learner for L0-Regularized Best Subset Selection
#'
#' @name LearnerRegrL0Learn
#' @aliases mlr_learners_regr.L0Learn
#'
#' @description
#' Sparse linear regression with L0, L0L1 or L0L2 penalties, fitted by
#' coordinate descent and local combinatorial search with
#' [L0Learn::L0Learn.cvfit()].
#' The penalty parameters `lambda` and `gamma` are chosen by cross-validation
#' inside `L0Learn::L0Learn.cvfit()`, then the pair with the smallest mean
#' cross-validated error is stored in the model and used for prediction.
#'
#' The loss is always the squared error.
#' Features must be `"integer"` or `"numeric"`, and missing values are not
#' supported by L0Learn.
#'
#' @section Initial parameter values:
#' All parameters have the tag `"train"` and are passed to
#' [L0Learn::L0Learn.cvfit()].
#' Parameters that are not set explicitly keep the defaults of L0Learn.
#' `nGamma`, `gammaMax` and `gammaMin` only matter for the `"L0L1"` and
#' `"L0L2"` penalties.
#'
#' @section Cross-validation folds:
#' `L0Learn::L0Learn.cvfit()` fails with an out-of-bounds error on some very
#' small data sets, for example `nFolds = 10` on the 32 rows of
#' `mlr3::tsk("mtcars")`.
#' Use a smaller `nFolds` such as `5L` when the number of rows is small.
#'
#' @section Model:
#' After training, `learner$model` is a list with three elements:
#' * `fit`: the `"L0LearnCV"` object returned by [L0Learn::L0Learn.cvfit()].
#' * `lambda`, `gamma`: the pair that minimizes the mean cross-validated error.
#'
#' @section Selected features:
#' `learner$selected_features()` returns the names of the features with a
#' non-zero coefficient at the selected `(lambda, gamma)` pair.
#'
#' @references
#' Hazimeh H, Mazumder R (2020).
#' "Fast Best Subset Selection: Coordinate Descent and Local Combinatorial
#' Optimization Algorithms."
#' *Operations Research*, 68(5), 1517-1537.
#' \doi{10.1287/opre.2019.1919}
#'
#' @examples
#' library(mlr3L0Learn)
#' learner <- mlr3::lrn("regr.L0Learn", penalty = "L0", nFolds = 5L)
#' task <- mlr3::tsk("mtcars")
#' learner$train(task)
#' learner$selected_features()
#' prediction <- learner$predict(task)
#' prediction$score(mlr3::msr("regr.mse"))
#'
#' @export
LearnerRegrL0Learn <- R6::R6Class(
  "LearnerRegrL0Learn",
  inherit = mlr3::LearnerRegr,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      param_set <- paradox::ps(
        penalty = paradox::p_fct(
          c("L0", "L0L1", "L0L2"), default = "L0", tags = "train"),
        algorithm = paradox::p_fct(
          c("CD", "CDPSI"), default = "CD", tags = "train"),
        maxSuppSize = paradox::p_int(
          lower = 1L, default = 100L, tags = "train"),
        nLambda = paradox::p_int(lower = 1L, default = 100L, tags = "train"),
        nGamma = paradox::p_int(lower = 1L, default = 10L, tags = "train"),
        gammaMax = paradox::p_dbl(lower = 0, default = 10, tags = "train"),
        gammaMin = paradox::p_dbl(lower = 0, default = 1e-4, tags = "train"),
        partialSort = paradox::p_lgl(default = TRUE, tags = "train"),
        maxIters = paradox::p_int(lower = 1L, default = 200L, tags = "train"),
        rtol = paradox::p_dbl(lower = 0, default = 1e-6, tags = "train"),
        atol = paradox::p_dbl(lower = 0, default = 1e-9, tags = "train"),
        activeSet = paradox::p_lgl(default = TRUE, tags = "train"),
        activeSetNum = paradox::p_int(
          lower = 1L, default = 3L, tags = "train"),
        maxSwaps = paradox::p_int(lower = 1L, default = 100L, tags = "train"),
        scaleDownFactor = paradox::p_dbl(
          lower = 0, upper = 1, default = 0.8, tags = "train"),
        screenSize = paradox::p_int(
          lower = 1L, default = 1000L, tags = "train"),
        nFolds = paradox::p_int(lower = 2L, default = 10L, tags = "train"),
        seed = paradox::p_int(default = 1L, tags = "train"),
        intercept = paradox::p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = "regr.L0Learn",
        packages = "L0Learn",
        feature_types = c("integer", "numeric"),
        predict_types = "response",
        param_set = param_set,
        properties = "selected_features",
        label = "L0-Regularized Regression",
        man = "mlr3L0Learn::LearnerRegrL0Learn"
      )
    },

    #' @description
    #' Returns the names of the features with a non-zero coefficient at the
    #' selected `(lambda, gamma)` pair.
    #' @return `character()`.
    selected_features = function() {
      if (is.null(self$model)) {
        stop("Learner has not been trained yet.")
      }
      beta <- l0learn_coef(self$model)
      self$state$feature_names[beta != 0]
    }
  ),

  private = list(
    .train = function(task) {
      x <- l0learn_matrix(task, task$feature_names)
      y <- task$truth()
      pv <- self$param_set$get_values(tags = "train")
      fit <- mlr3misc::invoke(
        L0Learn::L0Learn.cvfit,
        x = x, y = y, loss = "SquaredError",
        .args = pv)
      best_cv <- vapply(fit$cvMeans, min, numeric(1))
      g <- which.min(best_cv)
      cv_g <- as.numeric(fit$cvMeans[[g]])
      l <- which.min(cv_g)
      list(
        fit = fit,
        lambda = fit$fit$lambda[[g]][l],
        gamma = fit$fit$gamma[g])
    },

    .predict = function(task) {
      x <- l0learn_matrix(task, self$state$feature_names)
      pred <- stats::predict(
        self$model$fit, newx = x,
        lambda = self$model$lambda, gamma = self$model$gamma)
      list(response = as.numeric(pred))
    }
  )
)

l0learn_matrix <- function(task, feature_names) {
  x <- as.matrix(task$data(cols = feature_names))
  storage.mode(x) <- "double"
  x
}

l0learn_coef <- function(model) {
  beta <- stats::coef(
    model$fit, lambda = model$lambda, gamma = model$gamma)
  as.numeric(beta)[-1]
}
