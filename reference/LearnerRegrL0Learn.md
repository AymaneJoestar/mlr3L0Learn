# Regression Learner for L0-Regularized Best Subset Selection

Sparse linear regression with L0, L0L1 or L0L2 penalties, fitted by
coordinate descent and local combinatorial search with
[`L0Learn::L0Learn.cvfit()`](https://rdrr.io/pkg/L0Learn/man/L0Learn.cvfit.html).
The penalty parameters `lambda` and `gamma` are chosen by
cross-validation inside
[`L0Learn::L0Learn.cvfit()`](https://rdrr.io/pkg/L0Learn/man/L0Learn.cvfit.html),
then the pair with the smallest mean cross-validated error is stored in
the model and used for prediction.

The loss is always the squared error. Features must be `"integer"` or
`"numeric"`, and missing values are not supported by L0Learn.

## Initial parameter values

All parameters have the tag `"train"` and are passed to
[`L0Learn::L0Learn.cvfit()`](https://rdrr.io/pkg/L0Learn/man/L0Learn.cvfit.html),
except `selection`, which is used by the learner itself. Parameters that
are not set explicitly keep the defaults of L0Learn. `nGamma`,
`gammaMax` and `gammaMin` only matter for the `"L0L1"` and `"L0L2"`
penalties.

## Cross-validation folds

The folds are built by
[`L0Learn::L0Learn.cvfit()`](https://rdrr.io/pkg/L0Learn/man/L0Learn.cvfit.html),
which cannot split every number of rows `n` into every number of folds
`nFolds`. When `ceiling(n / nFolds) * (nFolds - 1) >= n` it either fails
with an out-of-bounds error or returns only `NaN` cross-validation
errors. For example `nFolds = 10` fails on the 32 rows of
`mlr3::tsk("mtcars")`. The learner then stops with an informative error,
and another value of `nFolds` must be chosen.

## Random numbers

[`L0Learn::L0Learn.cvfit()`](https://rdrr.io/pkg/L0Learn/man/L0Learn.cvfit.html)
calls `set.seed(seed)`. The learner restores the state of the random
number generator after training, so fitting it does not change the
random numbers drawn afterwards. The cross-validation folds are the same
for a given `seed`, whose default is `1`.

## Choice of lambda and gamma

With `selection = "min"` (the default) the learner uses the pair with
the smallest mean cross-validated error. When several values are tied,
the one furthest along the `lambda` path (the smallest `lambda`, usually
the largest support) is used.

With `selection = "1se"` the learner keeps the `gamma` of that minimum,
then takes the model on the `lambda` path with the smallest support
among those whose mean cross-validated error is within one standard
error of the minimum. The standard error is the standard deviation of
the fold errors at the minimum divided by `sqrt(nFolds)`. If it is not
available the minimum is used, with a warning. This usually gives a
smaller support and a more stable choice of features.

## Bounds and custom penalty grids

`lows` and `highs` are single numbers that bound every coefficient, with
`lows <= 0 <= highs` and `lows < highs`. Bounds are not supported by
`algorithm = "CDPSI"`. `lambdaGrid` is a list of strictly decreasing
positive numeric vectors. For `penalty = "L0"` it must contain exactly
one vector, and for `"L0L1"` and `"L0L2"` it contains one vector per
value of `gamma`. When it is set, `nLambda` is ignored by L0Learn and
`nGamma` is replaced by the number of vectors. L0Learn can drop the last
values of a grid whose support exceeds `maxSuppSize`, so the stored path
may be shorter than the grid. `excludeFirstK` cannot be larger than the
number of features, because L0Learn would abort the R session.

## Model

After training, `learner$model` is a list with these elements:

- `fit`: the `"L0LearnCV"` object returned by
  [`L0Learn::L0Learn.cvfit()`](https://rdrr.io/pkg/L0Learn/man/L0Learn.cvfit.html).

- `gamma_index`, `lambda_index`: the position of the selected pair in
  the regularization path of `fit`.

- `lambda`, `gamma`: the selected values.

- `support_size`: the number of non-zero coefficients of the selected
  model.

- `cv_mean`, `cv_sd`: the mean and standard deviation, across folds, of
  the cross-validated error at the selected pair.

## Selected features

`learner$selected_features()` returns the names of the features with a
non-zero coefficient at the selected `(lambda, gamma)` pair.

## References

Hazimeh H, Mazumder R (2020). "Fast Best Subset Selection: Coordinate
Descent and Local Combinatorial Optimization Algorithms." *Operations
Research*, 68(5), 1517-1537.
[doi:10.1287/opre.2019.1919](https://doi.org/10.1287/opre.2019.1919)

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`mlr3::LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html)
-\> `LearnerRegrL0Learn`

## Methods

### Public methods

- [`LearnerRegrL0Learn$new()`](#method-LearnerRegrL0Learn-initialize)

- [`LearnerRegrL0Learn$selected_features()`](#method-LearnerRegrL0Learn-selected_features)

- [`LearnerRegrL0Learn$clone()`](#method-LearnerRegrL0Learn-clone)

Inherited methods

- [`mlr3::Learner$base_learner()`](https://mlr3.mlr-org.com/reference/Learner.html#method-base_learner)
- [`mlr3::Learner$configure()`](https://mlr3.mlr-org.com/reference/Learner.html#method-configure)
- [`mlr3::Learner$encapsulate()`](https://mlr3.mlr-org.com/reference/Learner.html#method-encapsulate)
- [`mlr3::Learner$format()`](https://mlr3.mlr-org.com/reference/Learner.html#method-format)
- [`mlr3::Learner$help()`](https://mlr3.mlr-org.com/reference/Learner.html#method-help)
- [`mlr3::Learner$predict()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict)
- [`mlr3::Learner$predict_newdata()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict_newdata)
- [`mlr3::Learner$print()`](https://mlr3.mlr-org.com/reference/Learner.html#method-print)
- [`mlr3::Learner$reset()`](https://mlr3.mlr-org.com/reference/Learner.html#method-reset)
- [`mlr3::Learner$train()`](https://mlr3.mlr-org.com/reference/Learner.html#method-train)
- [`mlr3::LearnerRegr$predict_newdata_fast()`](https://mlr3.mlr-org.com/reference/LearnerRegr.html#method-predict_newdata_fast)

------------------------------------------------------------------------

### `LearnerRegrL0Learn$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    LearnerRegrL0Learn$new()

------------------------------------------------------------------------

### `LearnerRegrL0Learn$selected_features()`

Returns the names of the features with a non-zero coefficient at the
selected `(lambda, gamma)` pair.

#### Usage

    LearnerRegrL0Learn$selected_features()

#### Returns

[`character()`](https://rdrr.io/r/base/character.html).

------------------------------------------------------------------------

### `LearnerRegrL0Learn$clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerRegrL0Learn$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
library(mlr3L0Learn)
learner <- mlr3::lrn("regr.L0Learn", penalty = "L0", nFolds = 5L)
task <- mlr3::tsk("mtcars")
learner$train(task)
learner$selected_features()
#> [1] "am"   "carb" "drat" "qsec" "wt"  
prediction <- learner$predict(task)
prediction$score(mlr3::msr("regr.mse"))
#> regr.mse 
#> 4.898502 
```
