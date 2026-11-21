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
#' [L0Learn::L0Learn.cvfit()], except `selection`, which is used by the learner
#' itself.
#' Parameters that are not set explicitly keep the defaults of L0Learn.
#' `nGamma`, `gammaMax` and `gammaMin` only matter for the `"L0L1"` and
#' `"L0L2"` penalties.
#'
#' @section Cross-validation folds:
#' The folds are built by `L0Learn::L0Learn.cvfit()`, which cannot split every
#' number of rows `n` into every number of folds `nFolds`.
#' When `ceiling(n / nFolds) * (nFolds - 1) >= n` it either fails with an
#' out-of-bounds error or returns only `NaN` cross-validation errors.
#' For example `nFolds = 10` fails on the 32 rows of `mlr3::tsk("mtcars")`.
#' The learner then stops with an informative error, and another value of
#' `nFolds` must be chosen.
#'
#' @section Random numbers:
#' `L0Learn::L0Learn.cvfit()` calls `set.seed(seed)`.
#' The learner restores the state of the random number generator after
#' training, so fitting it does not change the random numbers drawn afterwards.
#' The cross-validation folds are the same for a given `seed`, whose default
#' is `1`.
#'
#' @section Choice of lambda and gamma:
#' With `selection = "min"` (the default) the learner uses the pair with the
#' smallest mean cross-validated error.
#' When several values are tied, the one furthest along the `lambda` path
#' (the smallest `lambda`, usually the largest support) is used.
#'
#' With `selection = "1se"` the learner keeps the `gamma` of that minimum, then
#' takes the model on the `lambda` path with the smallest support among those
#' whose mean cross-validated error is within one standard error of the minimum.
#' The standard error is the standard deviation of the fold errors at the
#' minimum divided by `sqrt(nFolds)`.
#' If it is not available the minimum is used, with a warning.
#' This usually gives a smaller support and a more stable choice of features.
#'
#' @section Bounds and custom penalty grids:
#' `lows` and `highs` are single numbers that bound every coefficient, with
#' `lows <= 0 <= highs` and `lows < highs`.
#' Bounds are not supported by `algorithm = "CDPSI"`.
#' `lambdaGrid` is a list of strictly decreasing positive numeric vectors.
#' For `penalty = "L0"` it must contain exactly one vector, and for `"L0L1"`
#' and `"L0L2"` it contains one vector per value of `gamma`.
#' When it is set, `nLambda` is ignored by L0Learn and `nGamma` is replaced by
#' the number of vectors.
#' L0Learn can drop the last values of a grid whose support exceeds
#' `maxSuppSize`, so the stored path may be shorter than the grid.
#' `excludeFirstK` cannot be larger than the number of features, because
#' L0Learn would abort the R session.
#'
#' @section Model:
#' After training, `learner$model` is a list with these elements:
#' * `fit`: the `"L0LearnCV"` object returned by [L0Learn::L0Learn.cvfit()].
#' * `gamma_index`, `lambda_index`: the position of the selected pair in the
#'   regularization path of `fit`.
#' * `lambda`, `gamma`: the selected values.
#' * `support_size`: the number of non-zero coefficients of the selected model.
#' * `cv_mean`, `cv_sd`: the mean and standard deviation, across folds, of the
#'   cross-validated error at the selected pair.
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
        nGamma = paradox::p_int(
          lower = 1L, default = 10L, tags = "train",
          depends = penalty %in% c("L0L1", "L0L2")),
        gammaMax = paradox::p_dbl(
          lower = 0, default = 10, tags = "train",
          depends = penalty %in% c("L0L1", "L0L2")),
        gammaMin = paradox::p_dbl(
          lower = 0, default = 1e-4, tags = "train",
          depends = penalty %in% c("L0L1", "L0L2")),
        partialSort = paradox::p_lgl(default = TRUE, tags = "train"),
        maxIters = paradox::p_int(lower = 1L, default = 200L, tags = "train"),
        rtol = paradox::p_dbl(
          lower = 0, upper = 1 - 1e-8, default = 1e-6, tags = "train"),
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
        excludeFirstK = paradox::p_int(
          lower = 0L, default = 0L, tags = "train"),
        lows = paradox::p_dbl(upper = 0, default = -Inf, tags = "train"),
        highs = paradox::p_dbl(lower = 0, default = Inf, tags = "train"),
        lambdaGrid = paradox::p_uty(
          default = list(), tags = "train", custom_check = l0learn_check_grid),
        selection = paradox::p_fct(
          c("min", "1se"), default = "min", tags = "train"),
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
        mlr3misc::stopf("Learner has not been trained yet.")
      }
      self$state$feature_names[l0learn_coef(self$model)$beta != 0]
    }
  ),

  private = list(
    .train = function(task) {
      x <- l0learn_matrix(task, task$feature_names)
      y <- task$truth()
      pv <- self$param_set$get_values(tags = "train")
      selection <- if (is.null(pv$selection)) "min" else pv$selection
      pv$selection <- NULL
      pv <- l0learn_prepare_args(pv, ncol(x))
      fit <- l0learn_cvfit(x, y, pv)
      l0learn_select(fit, selection, l0learn_n_folds(pv), nrow(x))
    },

    .predict = function(task) {
      x <- l0learn_matrix(task, self$state$feature_names)
      coefs <- l0learn_coef(self$model)
      list(response = as.numeric(coefs$a0 + x %*% coefs$beta))
    }
  )
)
