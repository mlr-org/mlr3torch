#' @import paradox
#' @import checkmate
#' @import data.table
#' @import mlr3misc
#' @importFrom R6 R6Class is.R6
#' @importFrom methods formalArgs
#' @importFrom utils getFromNamespace
#' @import torch
#' @import mlr3pipelines
#' @import mlr3
#' @import vctrs
#' @importFrom tools R_user_dir
#' @importFrom rlang !!
#'
#' @section Options:
#' * `mlr3torch.cache`:
#'   Whether to cache the downloaded data (`TRUE`) or not (`FALSE`, default).
#'   This can also be set to a specific folder on the file system to be used as the cache directory.
#'
"_PACKAGE"

# to silence RCMD check
utils::globalVariables(c("self", "private", "super", ".."))

mlr3torch_pipeops = new.env()
mlr3torch_learners = new.env()
mlr3torch_tasks = new.env()
mlr3torch_resamplings = new.env()
mlr3torch_pipeop_tags = c("torch", "activation")
mlr3torch_feature_types = c(img = "imageuri", lt = "lazy_tensor")

register_po = function(name, constructor, metainf = NULL) {
  if (name %in% names(mlr3torch_pipeops)) stopf("pipeop %s registered twice", name)
  mlr3torch_pipeops[[name]] = list(constructor = constructor, metainf = substitute(metainf))
}

register_resampling = function(name, constructor) {
  if (name %in% names(mlr3torch_resamplings)) stopf("resampling %s registered twice", name)
  mlr3torch_resamplings[[name]] = constructor
}

register_learner = function(name, constructor) {
  assert_class(constructor, "R6ClassGenerator")
  task_type = if (startsWith(name, "classif")) "classif" else "regr"
  # What I am doing here:
  # The problem is that we wan't to set the task_type when creating the learner from the dictionary
  # The initial idea was to add functions function(...) LearnerClass$new(..., task_type = "<task-type>")
  # This did not work because mlr3misc does not work with ... arguments (... arguments are not
  # passed further to the initialize method)
  # For this reason, we need this hacky solution here, might change in the future in mlr3misc
  fn = crate(function() {
    invoke(constructor$new, task_type = task_type, .args = as.list(match.call()[-1]))
  }, constructor, task_type, .parent = topenv())
  fmls = formals(constructor$public_methods$initialize)
  fmls$task_type = NULL
  formals(fn) = fmls
  if (name %in% names(mlr3torch_learners)) stopf("learner %s registered twice", name)
  mlr3torch_learners[[name]] = fn
}

register_task = function(name, constructor) {
  if (name %in% names(mlr3torch_tasks)) stopf("task %s registered twice", name)
  mlr3torch_tasks[[name]] = constructor
}

register_mlr3 = function() {
  mlr_learners = utils::getFromNamespace("mlr_learners", ns = "mlr3")
  iwalk(as.list(mlr3torch_learners), function(l, nm) mlr_learners$add(nm, l)) # nolint

  mlr_tasks = mlr3::mlr_tasks
  iwalk(as.list(mlr3torch_tasks), function(task, nm) mlr_tasks$add(nm, task)) # nolint

  mlr_resamplings = mlr3::mlr_resamplings
  iwalk(as.list(mlr3torch_resamplings), function(resampling, nm) mlr_resamplings$add(nm, resampling))

  mlr_reflections = utils::getFromNamespace("mlr_reflections", ns = "mlr3") # nolint
  iwalk(as.list(mlr3torch_feature_types), function(ft, nm) mlr_reflections$task_feature_types[[nm]] = ft) # nolint

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
      "on_epoch_end",
      "on_end"
    )
  )

}

register_mlr3pipelines = function() {
  mlr_pipeops = utils::getFromNamespace("mlr_pipeops", ns = "mlr3pipelines")
  iwalk(as.list(mlr3torch_pipeops), function(value, name) {
    # metainf is quoted by pipelines
    add = mlr_pipeops$add
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
  lg = lgr::get_logger(pkgname)
  assign("lg", lg, envir = parent.env(environment()))
  f = function(event) {
    event$msg = paste0("[mlr3torch] ", event$msg)
    TRUE
  }
  lg$set_filters(list(f))
  if (Sys.getenv("IN_PKGDOWN") == "true") {
    lg$set_threshold("warn")
  }

  register_namespace_callback(pkgname, "mlr3", register_mlr3)
  register_namespace_callback(pkgname, "mlr3pipelines", register_mlr3pipelines)

  assign("lg", lgr::get_logger(pkgname), envir = parent.env(environment()))
  if (Sys.getenv("IN_PKGDOWN") == "true") {
    lg$set_threshold("warn")
  }

  if (Sys.getenv("DEVTOOLS_LOAD", unset = "") == "mlr3torch") {
    # When loading mlr3torch with devtools, we always get a warning when hashing lazy tensors.
    # mlr3misc's calculate_hash function calls digest::digest which in turn serializes objects.
    # When mlr3torch is loaded with devtools the `$initialize()` method of the dataset that underpins the lazy tensor
    # behaves differently than when loaded with library() and always throws the warning:
    # "package:mlr3torch" might not be avaiable when loading.
    #
    # Because this is quite annoying, we here suppress warnings that arise from the calculate_hash function.

    fn = force(mlr3misc::calculate_hash)
    calculate_hash = mlr3misc::crate(function(...) {
      suppressWarnings(fn(...))
    }, .parent = getNamespace("mlr3misc"), fn)

    unlockBinding("calculate_hash", parent.env(getNamespace("mlr3torch")))
    assign("calculate_hash", calculate_hash, envir = parent.env(getNamespace("mlr3torch")))
    lockBinding("calculate_hash", parent.env(getNamespace("mlr3torch")))
  }

}

.onUnload = function(libPaths) { # nolint
  walk(names(mlr3torch_learners), function(nm) mlr_learners$remove(nm))
  walk(names(mlr3torch_resamplings), function(nm) mlr_resamplings$remove(nm))
  walk(names(mlr3torch_tasks), function(nm) mlr_tasks$remove(nm))
  walk(names(mlr3torch_pipeops), function(nm) mlr_pipeops$remove(nm))
  mlr_reflections$pipeops$valid_tags = setdiff(mlr_reflections$pipeops$valid_tags, mlr3torch_pipeop_tags)
  mlr_reflections$learner_feature_types = setdiff(mlr_reflections$learner_feature_types, mlr3torch_feature_types)
}

leanify_package()
