#' @import paradox
#' @import checkmate
#' @import data.table
#' @import mlr3misc
#' @importFrom R6 R6Class is.R6
#' @importFrom methods formalArgs
#' @importFrom utils getFromNamespace capture.output head tail
#' @import torch
#' @import mlr3pipelines
#' @import mlr3
#' @importFrom tools R_user_dir
#' @importFrom withr with_seed
#'
#' @section Options:
#' * `mlr3torch.cache`:
#'   Whether to cache the downloaded data (`TRUE`) or not (`FALSE`, default).
#'   This can also be set to a specific folder on the file system to be used as the cache directory.
#'
"_PACKAGE"

# to silence RCMD check
utils::globalVariables(c("self", "private", "super", ".."))
if (FALSE) knitr::knit2pandoc
if (FALSE) withr::with_seed

mlr3torch_task_generators = new.env()
mlr3torch_pipeop_tags = c("torch", "activation")
mlr3torch_feature_types = c(lt = "lazy_tensor")

# silence static checker
withr::with_seed

register_mlr3 = function() {
  mlr_learners = utils::getFromNamespace("mlr_learners", ns = "mlr3")
  iwalk(as.list(mlr3torch_learners), function(x, nm) mlr_learners$add(nm, x$fn, .prototype_args = x$prototype_args))

  mlr_tasks = mlr3::mlr_tasks
  iwalk(as.list(mlr3torch_tasks), function(task, nm) mlr_tasks$add(nm, task)) # nolint

  mlr_reflections = utils::getFromNamespace("mlr_reflections", ns = "mlr3") # nolint
  iwalk(as.list(mlr3torch_feature_types), function(ft, nm) mlr_reflections$task_feature_types[[nm]] = ft) # nolint
  mlr_reflections$loaded_packages = c(mlr_reflections$loaded_packages, "mlr3torch")

  mlr_reflections$torch = list(
    devices = c("auto", "cpu", "cuda", "mkldnn", "opengl", "opencl", "ideep", "hip", "fpga", "xla", "mps", "meta"),
    callback_stages = c(
      "on_begin",
      "on_epoch_begin",
      "on_batch_begin",
      "on_after_backward",
      "on_batch_end",
      "on_before_valid",
      "on_batch_valid_begin",
      "on_batch_valid_end",
      "on_valid_end",
      "on_epoch_end",
      "on_end",
      "on_exit"
    )
  )

}

register_mlr3pipelines = function() {
  mlr_reflections = utils::getFromNamespace("mlr_reflections", ns = "mlr3")
  mlr_pipeops = utils::getFromNamespace("mlr_pipeops", ns = "mlr3pipelines")
  add = mlr_pipeops$add # nolint
  iwalk(as.list(mlr3torch_pipeops), function(value, name) {
    # metainf is quoted by pipelines
    eval(call("add", quote(name), quote(value$constructor), value$metainf))
  })
  mlr_reflections$pipeops$valid_tags = unique(c(mlr_reflections$pipeops$valid_tags, mlr3torch_pipeop_tags))
  lapply(mlr3torch_pipeops, eval)
}

.onLoad = function(libname, pkgname) { # nolint
  # For caching directory
  backports::import(pkgname)
  backports::import(pkgname, "R_user_dir", force = TRUE)

  # Configure Logger:
  assign("lg", lgr::get_logger("mlr3"), envir = parent.env(environment()))
  if (Sys.getenv("IN_PKGDOWN") == "true") {
    lg$set_threshold("warn")
  }

  register_namespace_callback(pkgname, "mlr3", register_mlr3)
  register_namespace_callback(pkgname, "mlr3pipelines", register_mlr3pipelines)
}


.onUnload = function(libPaths) { # nolint
  walk(names(mlr3torch_learners), function(nm) mlr_learners$remove(nm))
  walk(names(mlr3torch_tasks), function(nm) mlr_tasks$remove(nm))
  walk(names(mlr3torch_pipeops), function(nm) mlr_pipeops$remove(nm))
  mlr_reflections$pipeops$valid_tags = setdiff(mlr_reflections$pipeops$valid_tags, mlr3torch_pipeop_tags)
  mlr_reflections$learner_feature_types = setdiff(mlr_reflections$learner_feature_types, mlr3torch_feature_types)
}

leanify_package()
