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
#' @section torch License:
#'
#' Some parts of this R package - especially the documentation - have been copied or adapted from the R package
#' [torch] that comes under the MIT License:
#'
#' MIT License
#'
#' Copyright (c) 2020 Daniel Falbel
#'
#' Permission is hereby granted, free of charge, to any person obtaining a copy
#' of this software and associated documentation files (the "Software"), to deal
#' in the Software without restriction, including without limitation the rights
#' to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#' copies of the Software, and to permit persons to whom the Software is
#' furnished to do so, subject to the following conditions:
#'
#' The above copyright notice and this permission notice shall be included in all
#' copies or substantial portions of the Software.
#'
#' THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#' IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#' FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#' AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#' LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#' OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#' SOFTWARE.
#'
"_PACKAGE"

# to silence RCMD check
utils::globalVariables(c("self", "private", "super"))



po_register_env = new.env()

register_po = function(name, constructor, metainf = NULL) {
  if (name %in% names(po_register_env)) stopf("pipeop %s registered twice", name)
  po_register_env[[name]] = constructor
}

register_mlr3 = function() {
  mlr_learners = utils::getFromNamespace("mlr_learners", ns = "mlr3")
  mlr_reflections = utils::getFromNamespace("mlr_reflections", ns = "mlr3")
  mlr_callbacks = utils::getFromNamespace("mlr_callbacks", ns = "mlr3misc")
  mlr_tasks = utils::getFromNamespace("mlr_tasks", ns = "mlr3")

  # Image URI feature (e.g. file path to .jpg etc.) for image classif tasks
  iwalk(mlr3torch_feature_types, function(ft, nm) mlr_reflections$task_feature_types[[nm]] = ft)
  iwalk(mlr3torch_callbacks, function(clbk, nm) mlr_callbacks$add(nm, clbk))
  iwalk(mlr3torch_tasks, function(task, nm) mlr_tasks$add(nm, task))
  mlr_tasks$add("tiny_imagenet", load_task_tiny_imagenet)
}

register_mlr3pipelines = function() {
  mlr_pipeops = utils::getFromNamespace("mlr_pipeops", ns = "mlr3pipelines")
  imap(as.list(po_register_env), function(value, name) mlr_pipeops$add(name, value))
  mlr_reflections$pipeops$valid_tags = unique(c(mlr_reflections$pipeops$valid_tags, c("torch", "activation")))

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

.onUnload = function(libPaths) {


}
