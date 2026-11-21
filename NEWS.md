# mlr3L0Learn 0.1.0

* New parameter `selection`: `"min"` (default) keeps the pair with the smallest
  cross-validated error, `"1se"` picks the sparsest model within one standard
  error of the minimum.
* The model now stores `support_size`, `cv_mean` and `cv_sd` for the selected
  pair.
* New parameters `lows`, `highs` and `lambdaGrid`, with explicit errors for
  combinations that L0Learn does not support.
* Errors from `L0Learn::L0Learn.cvfit()` are reported with their own message.
  The advice to change `nFolds` is only given for the fold splitting failure.

# mlr3L0Learn 0.0.1

* First version: `regr.L0Learn` learner, tests, benchmark vignette and
  documentation site.
