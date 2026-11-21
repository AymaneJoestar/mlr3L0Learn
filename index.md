# mlr3L0Learn

Documentation: <https://mlr3l0learn.netlify.app>

`mlr3L0Learn` adds a regression learner for best subset selection to the
[mlr3](https://mlr3.mlr-org.com) framework. It wraps
[`L0Learn::L0Learn.cvfit()`](https://rdrr.io/pkg/L0Learn/man/L0Learn.cvfit.html),
which fits sparse linear models with L0, L0L1 and L0L2 penalties using
coordinate descent and local combinatorial search ([Hazimeh and
Mazumder, 2020](https://doi.org/10.1287/opre.2019.1919)).

## Installation

``` r

remotes::install_github("AymaneJoestar/mlr3L0Learn")
```

## Usage

Loading the package registers the learner `regr.L0Learn` in the `mlr3`
learner dictionary.

``` r

library(mlr3)
library(mlr3L0Learn)

task <- tsk("mtcars")
learner <- lrn("regr.L0Learn", penalty = "L0", nFolds = 5L)
learner$train(task)
learner$selected_features()

prediction <- learner$predict(task)
prediction$score(msr("regr.mse"))
```

The penalty strength `lambda` (and `gamma` for the `"L0L1"` and `"L0L2"`
penalties) is chosen by cross-validation inside
[`L0Learn::L0Learn.cvfit()`](https://rdrr.io/pkg/L0Learn/man/L0Learn.cvfit.html).
The pair with the smallest cross-validated error is used for prediction.

[`L0Learn::L0Learn.cvfit()`](https://rdrr.io/pkg/L0Learn/man/L0Learn.cvfit.html)
cannot split every number of rows into every number of folds: it fails,
or returns only `NaN` errors, when
`ceiling(n / nFolds) * (nFolds - 1) >= n`. For example the default
`nFolds = 10` does not work on the 32 rows of `mtcars`. The learner
reports this with an informative error, so pass another `nFolds`.

Training does not change the global random number generator state. The
cross-validation folds are controlled by the `seed` parameter.

A benchmark against `regr.featureless`, `regr.cv_glmnet` and a tuned
k-nearest neighbors learner is in the vignette:

``` r

vignette("benchmark", package = "mlr3L0Learn")
```

## Related work

- [Course projects
  wiki](https://github.com/tdhock/2026-01-aa-grande-echelle/wiki/projets)
- [L0Learn](https://github.com/hazimehh/L0Learn), the package used for
  training and prediction
- [mlr3extralearners](https://github.com/mlr-org/mlr3extralearners),
  where this learner is meant to be proposed
- [mlr3learners](https://github.com/mlr-org/mlr3learners), which
  provides `regr.cv_glmnet` for L1 regularization
