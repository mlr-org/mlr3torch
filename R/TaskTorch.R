#' @title Generic Torch Task
#'
#' @name mlr_tasks_torch
#'
#' @description
#' A general-purpose [`Task`][mlr3::Task] for supervised learning problems that are neither
#' classification nor regression, such as multi-label classification or multi-output regression.
#' It may have any number of target columns, and nothing beyond that is assumed about the structure
#' of the problem.
#' For a problem without target columns -- an autoencoder, a masked or contrastive objective --
#' use [`TaskTorchUnsupervised`], which is the same class without the targets.
#'
#' Adding a proper task type to `mlr3` requires a `Task` class, a `Prediction` class, a handful of
#' `PredictionData` methods, a `Measure` and a number of entries in
#' [`mlr_reflections`][mlr3::mlr_reflections]; the *Adding a Custom Task Type* vignette walks through
#' this.
#' `TaskTorch` is the quick alternative: the task type `"torch"` is registered once by `mlr3torch`
#' and every custom problem is expressed as an *instance* of `TaskTorch` rather than as a new class.
#' What would otherwise be S3 methods dispatching on the task class are fields of the task instead,
#' and they are inferred from the types of the target columns when they are not given.
#'
#' The price is that all `TaskTorch` instances share a single task type, so `mlr3` cannot tell a
#' multi-label task apart from a multi-output regression task.
#' Learners and measures for different problems are therefore interchangeable as far as `mlr3` is
#' concerned, and mixing them up produces an error somewhere in `torch` instead of an error from
#' `mlr3`.
#' Use `TaskTorch` for experiments and one-off problems, and add a real task type when you package
#' up a learning problem for others.
#'
#' @section Inference: The three things `mlr3torch` needs to know about a task are the target tensor
#'   `y` of a batch ([`get_target_batchgetter()`]), the number of output units of the network
#'   ([`output_dim_for()`]) and how a network output becomes a prediction
#'   ([`encode_prediction()`]).
#'   All three are derived from the target columns:
#'
#'   | target columns | `y` | `output_dim` | `response` | `prob` |
#'   | --- | --- | --- | --- | --- |
#'   | one `factor` / `ordered` with `k` levels | `long` codes | `k` | `factor()` | `matrix()`, `k` columns |
#'   | one `numeric` / `integer` | `float` `(n, 1)` | 1 | `numeric()` | -- |
#'   | `d` `numeric` / `integer` | `float` `(n, d)` | `d` | `matrix()`, `d` columns | -- |
#'   | one `logical` | `float` `(n, 1)` | 1 | `logical()` | `numeric()` |
#'   | `d` `logical` | `float` `(n, d)` | `d` | `matrix()`, `d` columns | `matrix()`, `d` columns |
#'
#'   Note that a two-level `factor` target is *not* treated as binary classification: it gets two
#'   output units and is trained with a cross-entropy loss, like any other `factor` target.
#'   Encode the target as `logical` to get the single-logit variant.
#'
#'   Any of the three can be overwritten by passing `target_batchgetter`, `output_dim` or
#'   `prediction_encoder` to the constructor, which is also the only way to use a combination of
#'   target columns that is not in the table above.
#'
#'   What the task specifies is the *default*, and a learner has the last word on two of the three:
#'   [`LearnerTorchModule`] (`lrn("torch.module")`) takes a `target_batchgetter` of its own, and any
#'   [`LearnerTorch`] can overwrite the private `$.encode_prediction()` method.
#'   This matters when the network and the loss expect a different encoding than the task's default,
#'   e.g. when training on one-hot encoded class labels.
#'
#' @section Scoring: `$truth()` returns the target column itself if there is exactly one target and a
#'   [`data.table`][data.table::data.table] with one column per target if there are several.
#'   [`msr_torch()`] turns a plain R function of `truth` and `response` into a [`Measure`][mlr3::Measure]
#'   that scores such a prediction.
#'
#'   A measure that is passed as `measure` becomes the task's default measure, which is what
#'   `$score()` and `$aggregate()` use when they are called without arguments.
#'
#' @param id (`character(1)`)\cr
#'   The id of the task.
#' @param backend ([`DataBackend`][mlr3::DataBackend] or `data.frame()`)\cr
#'   The data.
#' @param target (`character()`)\cr
#'   The names of the target columns. At least one, see [`TaskTorchUnsupervised`] for a task
#'   without targets.
#' @param label (`character(1)`)\cr
#'   The label of the task.
#' @param target_batchgetter (`function()` or `NULL`)\cr
#'   Converts the target columns of a batch into the target tensor `y`.
#'   Takes an argument `data`, a [`data.table`][data.table::data.table] with only the target columns,
#'   and optionally an argument `x`, the named list of feature tensors of the batch.
#'   If `NULL` (default), it is inferred from the target column types.
#' @param output_dim (`integer(1)` or `NULL`)\cr
#'   The number of output units the network needs.
#'   If `NULL` (default), it is inferred from the target column types.
#' @param prediction_encoder (`function()` or `NULL`)\cr
#'   Converts the network output into a prediction.
#'   Takes the arguments `task`, `predict_tensor` and `predict_type` and returns a named `list()`
#'   with elements `response` and, optionally, `prob`.
#'   If `NULL` (default), it is inferred from the target column types.
#' @param measure ([`Measure`][mlr3::Measure] or `NULL`)\cr
#'   The default measure of the task, see section *Scoring*.
#'
#' @family Task
#' @seealso [`TaskTorchUnsupervised`]
#' @export
#' @examplesIf torch::torch_is_installed()
#' # multi-label classification: one logical column per label
#' d = data.frame(x1 = rnorm(50), x2 = rnorm(50))
#' d$a = d$x1 > 0
#' d$b = d$x2 > 0
#' task = as_task_torch(d, target = c("a", "b"), id = "labels")
#' task
#' output_dim_for(task)
TaskTorch = R6Class("TaskTorch",
  inherit = TaskSupervised,
  public = list(
    #' @field target_batchgetter (`function()` or `NULL`)\cr
    #'   See the construction argument.
    target_batchgetter = NULL,
    #' @field prediction_encoder (`function()` or `NULL`)\cr
    #'   See the construction argument.
    prediction_encoder = NULL,
    #' @field measure ([`Measure`][mlr3::Measure] or `NULL`)\cr
    #'   See the construction argument.
    measure = NULL,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, backend, target, label = NA_character_,
      target_batchgetter = NULL, output_dim = NULL, prediction_encoder = NULL, measure = NULL) {
      assert_character(target, any.missing = FALSE)
      if (!length(target)) {
        stopf("A TaskTorch is supervised and needs at least one target column, use TaskTorchUnsupervised for a task without one.") # nolint
      }
      super$initialize(id = id, task_type = "torch", backend = backend, target = target, label = label)
      task_torch_init_fields(self, private, target_batchgetter, output_dim, prediction_encoder, measure)
    },
    #' @description
    #' The ground truth, see section *Scoring*.
    #' @param rows (`integer()`)\cr
    #'   The rows to return the truth for. All rows if `NULL`.
    truth = function(rows = NULL) {
      target = self$target_names
      data = self$data(rows, cols = target)
      if (length(target) == 1L) data[[1L]] else data
    }
  ),
  active = list(
    #' @field hash (`character(1)`)\cr
    #'   The hash of the task.
    #'   In addition to what [`Task`][mlr3::Task] hashes, this covers the fields that define the
    #'   learning problem, so that two tasks over the same data but with different encodings do not
    #'   collide.
    hash = function(rhs) {
      assert_ro_binding(rhs)
      task_torch_hash(self, super$hash, private$.output_dim)
    },
    #' @field output_dim (`integer(1)`)\cr
    #'   The number of output units the network needs, see section *Inference*.
    output_dim = function(rhs) {
      if (!missing(rhs)) {
        private$.output_dim = assert_int(rhs, lower = 1L, coerce = TRUE)
        return(invisible(NULL))
      }
      task_torch_output_dim(self, private$.output_dim)
    }
  ),
  private = list(
    .output_dim = NULL,
    deep_clone = function(name, value) {
      if (name == "measure" && !is.null(value)) value$clone(deep = TRUE) else super$deep_clone(name, value)
    }
  )
)

#' @title Generic Unsupervised Torch Task
#'
#' @name mlr_tasks_torch_unsupervised
#'
#' @description
#' The unsupervised counterpart of [`TaskTorch`]: a general-purpose [`Task`][mlr3::Task] without
#' target columns, for learning problems such as autoencoders, denoising or masked objectives and
#' contrastive pretraining.
#' It shares the task type `"torch"` with [`TaskTorch`], so the same learners, pipeops and measures
#' work for both.
#'
#' Because there are no target columns, there is nothing to infer the learning problem from, and
#' `output_dim` and `prediction_encoder` have to be given -- see section *Inference* of
#' [`TaskTorch`] for what they do.
#' The batches of such a task have no `y` element by default, so the loss is called as
#' `loss(y_hat, NULL)` and has to ignore its second argument.
#'
#' @section Targets that are a Function of the Input:
#' If the target of a batch is a function of its *input* -- an autoencoder reconstructing its input,
#' a denoising or masked objective, contrastive pretraining -- pass a `target_batchgetter` that
#' declares an `x` argument, which receives the named list of feature tensors of the batch:
#'
#' ```
#' as_task_torch(data, output_dim = ncol(data),
#'   target_batchgetter = function(data, x) x[[1L]],
#'   prediction_encoder = function(task, predict_tensor, predict_type) {
#'     list(response = as.matrix(predict_tensor$cpu()))
#'   })
#' ```
#'
#' @section Scoring: `$truth()` returns `NULL`, so measures of such a task read the ground truth
#'   from the task itself, which [`msr_torch()`] arranges for any function that declares a `task`
#'   argument.
#'
#'   A measure that is passed as `measure` becomes the task's default measure, which is what
#'   `$score()` and `$aggregate()` use when they are called without arguments.
#'
#' @param id (`character(1)`)\cr
#'   The id of the task.
#' @param backend ([`DataBackend`][mlr3::DataBackend] or `data.frame()`)\cr
#'   The data.
#' @param label (`character(1)`)\cr
#'   The label of the task.
#' @param target_batchgetter (`function()` or `NULL`)\cr
#'   Defines the target tensor `y` of a batch.
#'   Takes an argument `data`, a [`data.table`][data.table::data.table] which is empty here, and
#'   optionally an argument `x`, the named list of feature tensors of the batch.
#'   If `NULL` (default), the batches have no `y` element.
#' @param output_dim (`integer(1)` or `NULL`)\cr
#'   The number of output units the network needs.
#' @param prediction_encoder (`function()` or `NULL`)\cr
#'   Converts the network output into a prediction.
#'   Takes the arguments `task`, `predict_tensor` and `predict_type` and returns a named `list()`
#'   with elements `response` and, optionally, `prob`.
#' @param measure ([`Measure`][mlr3::Measure] or `NULL`)\cr
#'   The default measure of the task, see section *Scoring*.
#'
#' @family Task
#' @seealso [`TaskTorch`]
#' @export
#' @examplesIf torch::torch_is_installed()
#' # an autoencoder reconstructs its own input
#' d = data.frame(x1 = rnorm(50), x2 = rnorm(50))
#' task = as_task_torch(d, id = "ae", output_dim = 2L,
#'   target_batchgetter = function(data, x) x[[1L]],
#'   prediction_encoder = function(task, predict_tensor, predict_type) {
#'     list(response = as.matrix(predict_tensor$cpu()))
#'   })
#' task
#' task$truth()
TaskTorchUnsupervised = R6Class("TaskTorchUnsupervised",
  inherit = TaskUnsupervised,
  public = list(
    #' @field target_batchgetter (`function()` or `NULL`)\cr
    #'   See the construction argument.
    target_batchgetter = NULL,
    #' @field prediction_encoder (`function()` or `NULL`)\cr
    #'   See the construction argument.
    prediction_encoder = NULL,
    #' @field measure ([`Measure`][mlr3::Measure] or `NULL`)\cr
    #'   See the construction argument.
    measure = NULL,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, backend, label = NA_character_, target_batchgetter = NULL,
      output_dim = NULL, prediction_encoder = NULL, measure = NULL) {
      super$initialize(id = id, task_type = "torch", backend = backend, label = label)
      task_torch_init_fields(self, private, target_batchgetter, output_dim, prediction_encoder, measure)
    },
    #' @description
    #' The ground truth, which an unsupervised task does not have, see section *Scoring*.
    #' @param rows (`integer()`)\cr
    #'   Ignored, a task without target columns has no ground truth.
    truth = function(rows = NULL) {
      NULL
    }
  ),
  active = list(
    #' @field hash (`character(1)`)\cr
    #'   The hash of the task.
    #'   In addition to what [`Task`][mlr3::Task] hashes, this covers the fields that define the
    #'   learning problem, so that two tasks over the same data but with different encodings do not
    #'   collide.
    hash = function(rhs) {
      assert_ro_binding(rhs)
      task_torch_hash(self, super$hash, private$.output_dim)
    },
    #' @field output_dim (`integer(1)`)\cr
    #'   The number of output units the network needs.
    output_dim = function(rhs) {
      if (!missing(rhs)) {
        private$.output_dim = assert_int(rhs, lower = 1L, coerce = TRUE)
        return(invisible(NULL))
      }
      task_torch_output_dim(self, private$.output_dim)
    }
  ),
  private = list(
    .output_dim = NULL,
    deep_clone = function(name, value) {
      if (name == "measure" && !is.null(value)) value$clone(deep = TRUE) else super$deep_clone(name, value)
    }
  )
)

# The fields that `TaskTorch` and `TaskTorchUnsupervised` have in common. They are validated and
# assigned in one place so that the two classes cannot drift apart.
task_torch_init_fields = function(self, private, target_batchgetter, output_dim, prediction_encoder, measure) { # nolint
  self$target_batchgetter = assert_function(target_batchgetter, args = "data", null.ok = TRUE)
  self$prediction_encoder = assert_function(prediction_encoder,
    args = c("task", "predict_tensor", "predict_type"), null.ok = TRUE)
  self$measure = assert_r6(measure, "Measure", null.ok = TRUE)
  private$.output_dim = assert_int(output_dim, lower = 1L, null.ok = TRUE, coerce = TRUE)
  invisible(NULL)
}

task_torch_hash = function(task, super_hash, output_dim) {
  calculate_hash(super_hash, output_dim, task$measure$hash,
    hash_input(task$target_batchgetter), hash_input(task$prediction_encoder))
}

task_torch_output_dim = function(task, output_dim) {
  if (!is.null(output_dim)) {
    return(output_dim)
  }
  spec = task_torch_spec(task)
  switch(spec$kind,
    factor = length(spec$levels),
    numeric = length(spec$cols),
    logical = length(spec$cols),
    stopf("Cannot infer the output dimension of task '%s' (%s), pass `output_dim` explicitly.", task$id, spec$why) # nolint
  )
}

#' @title Create a Generic Torch Task
#'
#' @description
#' Creates a task of the general-purpose task type of `mlr3torch` from a `data.frame()` or a
#' [`DataBackend`][mlr3::DataBackend]: a [`TaskTorch`] if `target` columns are given and a
#' [`TaskTorchUnsupervised`] if they are not.
#' See [`TaskTorch`] for what is inferred from the target columns and what the trade-offs are.
#'
#' @param x (`data.frame()` or [`DataBackend`][mlr3::DataBackend])\cr
#'   The data.
#' @param target (`character()`)\cr
#'   The names of the target columns. If empty, a [`TaskTorchUnsupervised`] is created.
#' @param id (`character(1)`)\cr
#'   The id of the task.
#' @param ... (any)\cr
#'   Further arguments passed to the task's `$new()`, such as `target_batchgetter`, `output_dim`,
#'   `prediction_encoder` or `measure`.
#' @return [`TaskTorch`] or [`TaskTorchUnsupervised`]
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
as_task_torch = function(x, target = character(0), id = deparse(substitute(x))[1L], ...) {
  target = assert_character(target, any.missing = FALSE, null.ok = TRUE) %??% character(0)
  id = assert_string(id)
  if (!length(target)) {
    return(TaskTorchUnsupervised$new(id = id, backend = x, ...))
  }
  TaskTorch$new(id = id, backend = x, target = target, ...)
}

# Describes what the target of a TaskTorch looks like. Everything that is inferred rather than
# passed to the constructor is derived from this.
task_torch_spec = function(task) {
  target = task$target_names
  if (!length(target)) {
    return(list(kind = "none", cols = character(0), why = "the task has no target columns"))
  }
  types = task$col_info[list(target), "type", on = "id"][[1L]]

  if (length(target) == 1L && types %in% c("factor", "ordered")) {
    levels = task$col_info[list(target), "levels", on = "id"][[1L]][[1L]]
    return(list(kind = "factor", cols = target, levels = levels))
  }
  if (all(types %in% c("numeric", "integer"))) {
    return(list(kind = "numeric", cols = target))
  }
  if (all(types == "logical")) {
    return(list(kind = "logical", cols = target))
  }
  list(kind = "unknown", cols = target,
    why = sprintf("the target columns have types %s", paste0("'", unique(types), "'", collapse = ", ")))
}

#' @export
output_dim_for.TaskTorch = function(x, ...) { # nolint
  x$output_dim
}

#' @export
output_dim_for.TaskTorchUnsupervised = function(x, ...) { # nolint
  x$output_dim
}

#' @export
get_target_batchgetter.TaskTorch = function(task, ...) { # nolint
  if (!is.null(task$target_batchgetter)) {
    return(task$target_batchgetter)
  }
  spec = task_torch_spec(task)
  switch(spec$kind,
    factor = function(data) torch_tensor(as.integer(data[[1L]]), dtype = torch_long()),
    numeric = function(data) torch_tensor(as.matrix(data), dtype = torch_float()),
    logical = function(data) torch_tensor(1 * as.matrix(data), dtype = torch_float()),
    none = NULL,
    stopf("Cannot infer the target batchgetter of task '%s' (%s), pass `target_batchgetter` explicitly.", task$id, spec$why) # nolint
  )
}

#' @export
encode_prediction.TaskTorch = function(task, network_output, predict_type, ...) { # nolint
  if (!is.null(task$prediction_encoder)) {
    # the raw output is passed on, so a `prediction_encoder` is also how a network with more than
    # one head is encoded
    return(task$prediction_encoder(task = task, predict_tensor = network_output,
      predict_type = predict_type))
  }
  # the inferred encodings below all expect a single tensor
  predict_tensor = assert_single_head(network_output, task)
  spec = task_torch_spec(task)
  switch(spec$kind,
    factor = {
      # The levels come from the task we are predicting on, while the width of `predict_tensor` was
      # fixed when the network was built. If they disagree -- e.g. because the predict task was
      # `$droplevels()`ed -- assigning the levels onto the argmax indices would silently relabel
      # every observation, so this has to be an error.
      shape = predict_tensor$shape
      if (length(shape) != 2L || shape[2L] != length(spec$levels)) {
        stopf("Network output of shape (%s) is incompatible with the %i levels of target '%s' of task '%s'. Was the network trained on a task with different levels?", paste(shape, collapse = ", "), length(spec$levels), spec$cols, task$id) # nolint
      }
      response = as.integer(with_no_grad(predict_tensor$argmax(dim = 2L))$to(device = "cpu"))
      class(response) = "factor"
      levels(response) = spec$levels
      prob = if (predict_type == "prob") {
        prob = as.matrix(with_no_grad(nnf_softmax(predict_tensor, dim = 2L))$to(device = "cpu"))
        colnames(prob) = spec$levels
        prob
      }
      list(response = response, prob = prob)
    },
    numeric = {
      if (predict_type != "response") {
        stopf("Task '%s' has numeric targets, for which only predict_type 'response' is available.", task$id) # nolint
      }
      predict_tensor = with_no_grad(predict_tensor)$to(device = "cpu")
      # a network for a single target may emit either a (n, 1) or a (n) tensor
      if (length(spec$cols) == 1L) {
        return(list(response = as.numeric(predict_tensor)))
      }
      response = as.matrix(predict_tensor)
      colnames(response) = spec$cols
      list(response = response)
    },
    logical = {
      prob = as.matrix(with_no_grad(nnf_sigmoid(predict_tensor))$to(device = "cpu"))
      colnames(prob) = spec$cols
      response = prob > 0.5
      if (length(spec$cols) == 1L) {
        response = as.logical(response)
        prob = as.numeric(prob)
      }
      list(response = response, prob = if (predict_type == "prob") prob)
    },
    stopf("Cannot infer the prediction encoding of task '%s' (%s), pass `prediction_encoder` explicitly.", task$id, spec$why) # nolint
  )
}

#' @export
get_target_batchgetter.TaskTorchUnsupervised = function(task, ...) { # nolint
  # a task without target columns has nothing to infer from, so it either was given a
  # `target_batchgetter` or its batches have no `y` at all
  task$target_batchgetter
}

#' @export
encode_prediction.TaskTorchUnsupervised = function(task, network_output, predict_type, ...) { # nolint
  if (is.null(task$prediction_encoder)) {
    stopf("Task '%s' has no target columns, so its prediction encoding cannot be inferred, pass `prediction_encoder` explicitly.", task$id) # nolint
  }
  task$prediction_encoder(task = task, predict_tensor = network_output, predict_type = predict_type)
}
