#' @import paradox
#' @import checkmate
#' @import data.table
#' @import mlr3misc
#' @importFrom R6 R6Class is.R6
#' @import torch
#' @import mlr3pipelines
#' @import mlr3
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
utils::globalVariables(c("self", "private", "super"))

po_register_env = new.env()
register_po = function(name, constructor, metainf = NULL) {
  if (name %in% names(po_register_env)) stopf("pipeop %s registered twice", name)
  po_register_env[[name]] = substitute(mlr_pipeops$add(name, constructor, metainf))
}

register_mlr3 = function() {
  # Learners ----------------------------------------------------------------

  mlr_learners = utils::getFromNamespace("mlr_learners", ns = "mlr3")

  # Reflections -------------------------------------------------------------
  mlr_reflections = utils::getFromNamespace("mlr_reflections", ns = "mlr3")

  # Image URI feature (e.g. file path to .jpg etc.) for image classif tasks
  mlr_reflections$task_feature_types[["img"]] = "imageuri"
}

register_mlr3pipelines = function() {
  mlr_pipeops = utils::getFromNamespace("mlr_pipeops", ns = "mlr3pipelines")
  lapply(po_register_env, eval)
}

.onLoad = function(libname, pkgname) { # nolint
  # For caching directory
  backports::import(pkgname)
  backports::import(pkgname, "R_user_dir", force = TRUE)

  register_namespace_callback(pkgname, "mlr3", register_mlr3)
  register_namespace_callback(pkgname, "mlr3pipelines", register_mlr3pipelines)

  assign("lg", lgr::get_logger(pkgname), envir = parent.env(environment()))
  if (Sys.getenv("IN_PKGDOWN") == "true") {
    lg$set_threshold("warn")
  }
}
