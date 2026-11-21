.onLoad <- function(libname, pkgname) {
  mlr3misc::register_namespace_callback(pkgname, "mlr3", function() {
    mlr3::mlr_learners$add("regr.L0Learn", LearnerRegrL0Learn)
  })
}
