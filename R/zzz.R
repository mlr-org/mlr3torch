#' @import paradox
#' @import checkmate
#' @import data.table
#' @import mlr3misc
#' @importFrom R6 R6Class
#' @import torch
#' @import mlr3pipelines
#' @import mlr3misc
#' @import mlr3
#' @importFrom R6 R6Class is.R6
#' @importFrom zeallot `%<-%`
#' @importFrom coro loop
#' @importFrom methods formalArgs hasArg
#' @importFrom utils getFromNamespace
#' @importFrom progress progress_bar
#'
#' @description
#'   mlr3torch Connects the R `torch` package to mlr3.
#'   Neural Networks can be implemented on three different levels of control:
#'
#'   * Custom `nn_module`
#'   * Using `TorchOp`s
#'   * Predefined architectures via `Learner`s
#'
#' @section Feature Types:
#'   It adds the feature type "imageuri", which is a S3 class based on a character vector with
#'   additional attributes.
#'
"_PACKAGE"

# to silence RCMD check
utils::globalVariables(c("..", "self", "private", "super", "N", "..input"))


register_mlr3 = function() {
  # Learners ----------------------------------------------------------------

  lrns = utils::getFromNamespace("mlr_learners", ns = "mlr3")
  tsks = utils::getFromNamespace("mlr_tasks", ns = "mlr3")

  tsks$add("tiny_imagenet", load_task_tiny_imagenet)

  # classification learners
  ## lrns$add("classif.tabnet", LearnerClassifTabNet)
  ## lrns$add("classif.alexnet", LearnerClassifAlexNet)
  lrns$add("classif.torch", LearnerClassifTorch)
  ## lrns$add("classif.tab_resnet", LearnerClassifTabResNet)
  ## lrns$add("classif.mlp", LearnerClassifMLP)

  # regression learners
  lrns$add("regr.tabnet", LearnerRegrTabNet)

  # PipeOps
  mlr_pipeops = mlr3pipelines::mlr_pipeops
  mlr_pipeops$add("imagetrafo", PipeOpImageTrafo)


  # Reflections -------------------------------------------------------------
  reflcts = utils::getFromNamespace("mlr_reflections", ns = "mlr3")

  reflcts$learner_param_tags = c(reflcts$learner_param_tags, "network", "optimizer", "loss")

  # Image URI feature (e.g. file path to .jpg etc.) for image classif tasks
  reflcts$task_feature_types[["img"]] = "imageuri"
}

.onLoad = function(libname, pkgname) { # nolint
  # For caching directory
  backports::import(pkgname)
  backports::import(pkgname, "R_user_dir", force = TRUE)

  register_namespace_callback(pkgname, "mlr3", register_mlr3)
  assign("lg", lgr::get_logger(pkgname), envir = parent.env(environment()))
  if (Sys.getenv("IN_PKGDOWN") == "true") {
    lg$set_threshold("warn")
  }
}
