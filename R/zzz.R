#' import paradox
#' import R6
#' import torch
#' import luz
#' import mlr3pipelines
#' import mlr3
"_PACKAGE"


register_mlr3 = function() {

  # Learners ----------------------------------------------------------------

  lrns = utils::getFromNamespace("mlr_learners", ns = "mlr3")

  # classification learners
  lrns$add("classif.torch.tabnet", LearnerClassifTorchTabnet)


  # regression learners
  lrns$add("regr.torch.tabnet", LearnerRegrTorchTabnet)


  # Reflections -------------------------------------------------------------
  reflcts = utils::getFromNamespace("mlr_reflections", ns = "mlr3")

  # Image URI feature (e.g. file path to .jpg etc.) for image classif tasks
  reflcts$task_feature_types[["img"]] = "imageuri"

}

.onLoad = function(libname, pkgname) {
  register_namespace_callback(pkgname, "mlr3", register_mlr3)
}
