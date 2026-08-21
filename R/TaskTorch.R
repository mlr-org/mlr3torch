#' @title Generic Torch Task
#'
#' @name mlr_tasks_torch
#'
#' @description
#' A general-purpose [`Task`][mlr3::Task] that can be used for modeling arbitrary problems, including
#' supervised and unsupervised problems.
#' The article on *Custom Learning Problems* covers all of this in more detail.
#'
#' The problem this generic task solves is that it is rather complicated to register new task 
#' types with `mlr3`, so this class makes this easier.
#' The price of this flexibility is the loss of some compatibility checks.
#'
#' @template params_task_torch
#' @param backend ([`DataBackend`][mlr3::DataBackend] or `data.frame()`)\cr
#'   The data.
#' @param label (`character(1)`)\cr
#'   The label of the task.
#' @param output_dim (`function()` or `NULL`)\cr
#'   Returns the number of output units the network needs.
#'   Takes an argument `task` and returns a single positive integer.
#'   May be `NULL` (default), in which case any caller of [`output_dim_for()`] errors.
#' @param default_encoder (`function()` or `NULL`)\cr
#'   The default prediction encoder for the task. This can be overwritten by a learner's
#'   private `$.encode_prediction` method.
#'   See [`LearnerTorch`] for more information.
#' @param default_measure ([`Measure`][mlr3::Measure] or `NULL`)\cr
#'   The default measure of the task, i.e. what [`msr("torch.default")`][mlr_measures_torch.default]
#'   resolves to.
#'   `rr$score()` and `rr$aggregate()` of a [`ResampleResult`][mlr3::ResampleResult] use it without
#'   being told, and `prediction$score(msr("torch.default"), task = task)` uses it when the task is
#'   passed along -- a [`Prediction`][mlr3::Prediction] carries none, so `prediction$score()` on its
#'   own cannot resolve it.
#'
#' @family Task
#' @export
#' @examplesIf torch::torch_is_installed()
#' # multi-label classification: one logical column per label
#' d = data.frame(x1 = rnorm(50), x2 = rnorm(50))
#' d$a = d$x1 > 0
#' d$b = d$x2 > 0
#' task = as_task_torch(d, target = c("a", "b"), id = "labels",
#'   output_dim = function(task) length(task$target_names),
#'   default_encoder = function(task, network_output, predict_type) {
#'     prob = as.matrix(torch::nnf_sigmoid(network_output)$cpu())
#'     colnames(prob) = task$target_names
#'     list(response = prob > 0.5, prob = if (predict_type == "prob") prob)
#'   })
#' task
#' output_dim_for(task)
TaskTorch = R6Class("TaskTorch",
  inherit = Task,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, backend, target = NULL, label = NA_character_,
      output_dim = NULL, default_encoder = NULL, default_measure = NULL) {
      target = assert_character(target, any.missing = FALSE, unique = TRUE, min.len = 1L, null.ok = TRUE) %??% character(0)
      super$initialize(id = id, task_type = "torch", backend = backend, label = label)
      assert_subset(target, self$col_roles$feature)
      self$col_roles$target = target
      self$col_roles$feature = setdiff(self$col_roles$feature, target)

      self$output_dim = output_dim
      private$.default_encoder = assert_function(default_encoder,
        args = c("task", "network_output", "predict_type"), null.ok = TRUE)
      private$.default_measure = assert_r6(default_measure, "Measure", null.ok = TRUE)
    },
    #' @description
    #' The ground truth, see section *Scoring*.
    #' Might return `NULL` for unsupervised problems.
    #' @param rows (`integer()`)\cr
    #'   The rows to return the truth for. All rows if `NULL`.
    truth = function(rows = NULL) {
      target = self$target_names
      if (!length(target)) {
        return(NULL)
      }
      data = self$data(rows, cols = target)
      if (length(target) == 1L) data[[1L]] else data
    }
  ),
  active = list(
    #' @field hash (`character(1)`)\cr
    #'   The hash of the task.
    hash = function(rhs) {
      assert_ro_binding(rhs)
      calculate_hash(super$hash, self$default_measure$hash,
        private$.output_dim, self$default_encoder)
    },
    #' @field default_encoder (`function()` or `NULL`)\cr
    #'   The default prediction encoder. Read-only.
    default_encoder = function(rhs) {
      assert_ro_binding(rhs)
      private$.default_encoder
    },
    #' @field default_measure ([`Measure`][mlr3::Measure] or `NULL`)\cr
    #'   See the construction argument. Read-only, for the same reason as `default_encoder`.
    default_measure = function(rhs) {
      assert_ro_binding(rhs)
      private$.default_measure
    },
    #' @field output_dim (`function()` or `NULL`)\cr
    #'   See the construction argument.
    #'   Called by [`output_dim_for()`].
    output_dim = function(rhs) {
      if (missing(rhs)) {
        return(private$.output_dim)
      }
      private$.output_dim = assert_function(rhs, args = "task", null.ok = TRUE)
      invisible(NULL)
    }
  ),
  private = list(
    .output_dim = NULL,
    .default_encoder = NULL,
    .default_measure = NULL,
    deep_clone = function(name, value) {
      if (name == ".default_measure" && !is.null(value)) value$clone(deep = TRUE) else super$deep_clone(name, value)
    }
  )
)

#' @title Create a Generic Torch Task
#'
#' @description
#' Creates a [`TaskTorch`], the general-purpose task type of `mlr3torch`, from a `data.frame()` or a
#' [`DataBackend`][mlr3::DataBackend].
#' See the *Custom Learning Problems* article for more information.
#'
#' @param x (`data.frame()` or [`DataBackend`][mlr3::DataBackend])\cr
#'   The data.
#' @template params_task_torch
#' @param ... (any)\cr
#'   Further arguments passed to [`TaskTorch`]`$new()`, such as `output_dim`, `default_encoder`
#'   or `default_measure`.
#' @return [`TaskTorch`]
#' @export
#' @examplesIf torch::torch_is_installed()
#' # multi-output regression
#' d = data.frame(x = rnorm(50))
#' d$y1 = d$x + rnorm(50)
#' d$y2 = 2 * d$x + rnorm(50)
#' as_task_torch(d, target = c("y1", "y2"))
#'
#' # unsupervised: no target at all
#' as_task_torch(data.frame(a = rnorm(50), b = rnorm(50)))
as_task_torch = function(x, target = NULL, id = deparse(substitute(x))[1L], ...) {
  if (inherits(x, "TaskTorch")) {
    return(x)
  }
  if (inherits(x, "Task")) {
    stopf("Task '%s' is a <%s> and cannot be converted into a TaskTorch. Build one from its data instead, e.g. as_task_torch(x$data(), target = x$target_names, id = x$id), which takes the rows and columns the task currently has but none of its other row and column roles.", x$id, class(x)[[1L]]) # nolint
  }
  TaskTorch$new(id = assert_string(id), backend = x, target = target, ...)
}


#' @export
output_dim_for.TaskTorch = function(x, ...) { # nolint
  if (is.null(x$output_dim)) {
    stopf("Task '%s' has no `output_dim`. Pass one to the task, or size the network's output yourself with `nn(\"linear\")` instead of `nn(\"head\")`.", x$id) # nolint
  }
  assert_int(x$output_dim(task = x), lower = 1L, coerce = TRUE)
}

#' @export
get_target_batchgetter.TaskTorch = function(task, ...) { # nolint
  # A task with no target has no `y`: the loss is called as `loss(y_hat)` and there is nothing for a
  # batchgetter to build, so the learner does not have to pass one.
  if (!length(task$target_names)) {
    return(NULL)
  }
  stopf("Task '%s' does not define how its target becomes a tensor -- what `y` has to look like follows from the loss, so it is the learner that decides. Pass `target_batchgetter` to the learner (e.g. `lrn(\"torch.module\")`) or overwrite the method for your own `LearnerTorch` subclass.", task$id) # nolint
}

#' @export
encode_prediction.TaskTorch = function(task, network_output, predict_type, ...) { # nolint
  if (is.null(task$default_encoder)) {
    stopf("Task '%s' has no `default_encoder`, so there is no way to turn the network's output into a prediction. Pass one to the task, or overwrite the private `.encode_prediction()` method of the learner.", task$id) # nolint
  }
  task$default_encoder(task = task, network_output = network_output, predict_type = predict_type)
}
