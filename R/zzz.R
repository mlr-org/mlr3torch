#' @import paradox
#' @import checkmate
#' @import data.table
#' @import mlr3misc
#' @importFrom R6 R6Class is.R6
#' @importFrom methods formalArgs
#' @import torch
#' @import mlr3pipelines
#' @import mlr3
#'
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

mlr3torch_pipeops = new.env()
mlr3torch_learners = new.env()
mlr3torch_tasks = new.env()
mlr3torch_tags = c("torch", "activation")
mlr3torch_feature_types = list(img = "imageuri")

mlr3torch_activations = c(
  "celu",
  "tanh",
  "softpluts",
  "rrelu",
  "softsign",
  "relu",
  "sigmoid",
  "gelu",
  "hardtanh",
  "linear",
  "prelu",
  "relu6",
  "selu",
  "hardshrink",
  "softshrink",
  "leaky_relu"
)

# these are sorted in the order in which they are executed
mlr3torch_callback_stages = c(
  "on_begin",
  "on_epoch_begin",
  "on_batch_begin",
  "on_after_backward",
  "on_batch_end",
  "on_before_valid",
  "on_batch_valid_begin",
  "on_batch_valid_end",
  "on_epoch_end",
  "on_end"
)


# metainf must be manually added in the register_mlr3pipelines function
# Because the value is substituted, we cannot pass it through this function
register_po = function(name, constructor) {
  if (name %in% names(mlr3torch_pipeops)) stopf("pipeop %s registered twice", name)
  mlr3torch_pipeops[[name]] = list(constructor = constructor)
}

register_learner = function(name, constructor) {
  if (name %in% names(mlr3torch_learners)) stopf("learner %s registered twice", name)
  mlr3torch_learners[[name]] = constructor
}

register_task = function(name, constructor) {
  if (name %in% names(mlr3torch_tasks)) stopf("task %s registered twice", name)
  mlr3torch_tasks[[name]] = constructor
}

register_mlr3 = function() {
  mlr_learners = utils::getFromNamespace("mlr_learners", ns = "mlr3")
  iwalk(as.list(mlr3torch_learners), function(l, nm) mlr_learners$add(nm, l)) # nolint

  mlr_tasks = utils::getFromNamespace("mlr_tasks", ns = "mlr3")
  iwalk(as.list(mlr3torch_tasks), function(task, nm) mlr_tasks$add(nm, task)) # nolint

  mlr_reflections = utils::getFromNamespace("mlr_reflections", ns = "mlr3") # nolint
  iwalk(as.list(mlr3torch_feature_types), function(ft, nm) mlr_reflections$task_feature_types[[nm]] = ft) # nolint
}

register_mlr3pipelines = function() {
  mlr_pipeops = utils::getFromNamespace("mlr_pipeops", ns = "mlr3pipelines")
  iwalk(as.list(mlr3torch_pipeops), function(value, name) {
    mlr_pipeops$add(name, value$constructor, value$metainf)
  })
  mlr_pipeops$metainf$torch_loss = list(loss = t_loss("cross_entropy"))
  mlr_reflections$pipeops$valid_tags = unique(c(mlr_reflections$pipeops$valid_tags, c("torch", "activation")))
  lapply(mlr3torch_pipeops, eval)
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

# TODO: The removal of properties fails when the property has been alrady present before it was added in torch
# --> Take care that we don't add properties that are present
.onUnload = function(libPaths) { # nolint
  mlr_learners = utils::getFromNamespace("mlr_learners", ns = "mlr3")
  mlr_callbacks = utils::getFromNamespace("mlr_callbacks", ns = "mlr3misc")
  mlr_tasks = utils::getFromNamespace("mlr_tasks", ns = "mlr3")
  mlr_reflections = utils::getFromNamespace("mlr_reflections", ns = "mlr3") # nolint

  walk(mlr3torch_learners, function(nm) mlr_learners$remove(nm))
  walk(mlr3torch_callbacks, function(nm) mlr_callbacks$remove(nm))
  walk(mlr3torch_tasks, function(nm) mlr_tasks$remove(nm))
  walk(names(mlr3torch_feature_types), function(nm) mlr_reflections$task_feature_types[[nm]] = NULL)
  # walk(names(mlr3torch_learner_properties), function(nm) mlr_reflections$learner_properties[[nm]] = NULL)
}
