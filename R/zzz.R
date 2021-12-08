register_mlr3 = function() {
  x = utils::getFromNamespace("mlr_learners", ns = "mlr3")

  # classification learners
  x$add("classif.torch.tabnet", LearnerClassifTorchTabnet)


  # regression learners
  x$add("regr.torch.tabnet", LearnerRegrTorchTabnet)

}

.onLoad = function(libname, pkgname) {
  register_namespace_callback(pkgname, "mlr3", register_mlr3)
}
