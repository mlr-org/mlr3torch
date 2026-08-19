#' @title Generic Torch Task
#'
#' @name mlr_tasks_torch
#'
#' @description
#' A general-purpose [`Task`][mlr3::Task] for learning problems that are neither classification nor
#' regression, such as multi-label classification, multi-output regression or autoencoders.
#' It may have any number of target columns, including none at all, so it is supervised or
#' unsupervised depending only on whether target columns were given.
#' Nothing beyond that is assumed about the structure of the problem.
#'
#' It inherits from [`TaskSupervised`][mlr3::TaskSupervised] even when it has no target columns,
#' because that is what `mlr3` dispatches the ground truth of a prediction on: `as_prediction_data()`
#' copies `task$truth()` into the prediction data only for a `TaskSupervised`.
#' The one rule of that class which does not apply here, that there be at least one target column,
#' is lifted in `task_check_col_roles.TaskTorch()` below.
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
#' @section What you have to specify: `mlr3torch` needs to know three things about a learning
#'   problem, and a `TaskTorch` infers none of them, because the column types of a target say how it
#'   is stored, not how you want to model it. A two-level `factor` could be one logit or two; `d`
#'   numeric columns could be one head of width `d` or `d` separate heads. Guessing would be a
#'   modelling decision made on your behalf, so all three are stated explicitly:
#'
#'   | what | who provides it | how |
#'   | --- | --- | --- |
#'   | the target tensor `y` of a batch | the **learner** | `target_batchgetter` of [`LearnerTorchModule`] / [`LearnerTorchModel`], or [`get_target_batchgetter()`] for your own subclass |
#'   | the number of output units | the **task** | `output_dim`, read through [`output_dim_for()`] |
#'   | how a network output becomes a prediction | the **task** | `prediction_encoder`, read through [`encode_prediction()`] |
#'
#'   The split follows what each one is consumed by. What `y` has to look like is dictated by the
#'   loss, which belongs to the learner: cross entropy wants `long` class indices where mean squared
#'   error wants `float`, so the same task trained with two losses needs two different tensors.
#'   What a prediction looks like is what the task's measures are written against, so it belongs to
#'   the task.
#'
#'   The encoder is also where any consistency check between the network and the task belongs. A
#'   network trained on a task with three target levels emits three columns, and mapping those onto
#'   the levels of a task that has since been `$droplevels()`ed would relabel every observation
#'   rather than fail, so an encoder that assigns levels should verify the width it was given.
#'
#'   A learner always has the last word: [`LearnerTorch`] can overwrite the private
#'   `$.encode_prediction()` method, which is how [`LearnerTorchVision`] drops an auxiliary head and
#'   `lrn("classif.tabm")` averages over its ensemble dimension.
#'
#'   `output_dim` is a `function(task)` rather than a number so that it still holds after the task
#'   is changed -- a different target column, a `$droplevels()` -- and it is optional: a network
#'   that sizes its own output, such as one with several heads of different widths, never asks for
#'   it. `nn("head")` and any module calling [`output_dim_for()`] do ask, and error if it is unset.
#'
#' @section Tasks without a Target: A task may have no target columns at all.
#'   Its batches have no `y` element, so the loss is called as `loss(y_hat)`, with no second
#'   argument at all.
#'
#'   If the target of a batch is a function of its *input* rather than of a column -- an autoencoder
#'   reconstructing its input, a denoising or masked objective, contrastive pretraining -- give the
#'   learner a `target_batchgetter` that declares an `x` argument, which receives the named list of
#'   feature tensors of the batch:
#'
#'   ```
#'   task = as_task_torch(data, output_dim = function(task) ncol(data),
#'     prediction_encoder = function(task, predict_tensor, predict_type) {
#'       list(response = as.matrix(predict_tensor$cpu()))
#'     })
#'   lrn("torch.module", target_batchgetter = function(data, x) x[[1L]], ...)
#'   ```
#'
#'   Such a task has no `truth`, so its measures read the ground truth from the task, which
#'   [`msr_torch()`] arranges for any function that declares a `task` argument.
#'
#' @section Scoring: `$truth()` returns the target column itself if there is exactly one target, a
#'   [`data.table`][data.table::data.table] with one column per target if there are several, and
#'   `NULL` if the task has no target at all.
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
#'   The names of the target columns. May be empty.
#' @param label (`character(1)`)\cr
#'   The label of the task.
#' @param output_dim (`function()` or `NULL`)\cr
#'   Returns the number of output units the network needs.
#'   Takes an argument `task` and returns a single positive integer.
#'   May be `NULL` (default), in which case any caller of [`output_dim_for()`] errors.
#' @param prediction_encoder (`function()` or `NULL`)\cr
#'   Converts the network output into a prediction.
#'   Takes the arguments `task`, `predict_tensor` -- the network's output, unchanged, so possibly a
#'   `list()` of tensors -- and `predict_type`, and returns a named `list()` with elements `response`
#'   and, optionally, `prob`.
#'   May be `NULL` (default) if the learner encodes predictions itself.
#' @param measure ([`Measure`][mlr3::Measure] or `NULL`)\cr
#'   The default measure of the task, see section *Scoring*.
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
#'   prediction_encoder = function(task, predict_tensor, predict_type) {
#'     prob = as.matrix(torch::nnf_sigmoid(predict_tensor)$cpu())
#'     colnames(prob) = task$target_names
#'     list(response = prob > 0.5, prob = if (predict_type == "prob") prob)
#'   })
#' task
#' output_dim_for(task)
TaskTorch = R6Class("TaskTorch",
  inherit = TaskSupervised,
  public = list(
    prediction_encoder = NULL,
    #' @field measure ([`Measure`][mlr3::Measure] or `NULL`)\cr
    #'   See the construction argument.
    measure = NULL,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, backend, target = character(0), label = NA_character_,
      output_dim = NULL, prediction_encoder = NULL, measure = NULL) {
      target = assert_character(target, any.missing = FALSE, null.ok = TRUE) %??% character(0)
      super$initialize(id = id, task_type = "torch", backend = backend, target = target, label = label)

      self$output_dim = output_dim
      self$prediction_encoder = assert_function(prediction_encoder,
        args = c("task", "predict_tensor", "predict_type"), null.ok = TRUE)
      self$measure = assert_r6(measure, "Measure", null.ok = TRUE)
    },
    #' @description
    #' The ground truth, see section *Scoring*.
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
    #'   In addition to what [`Task`][mlr3::Task] hashes, this covers the fields that define the
    #'   learning problem, so that two tasks over the same data but with different encodings do not
    #'   collide.
    hash = function(rhs) {
      assert_ro_binding(rhs)
      calculate_hash(super$hash, self$measure$hash,
        hash_input(private$.output_dim), hash_input(self$prediction_encoder))
    },
    #' @field output_dim (`function()` or `NULL`)\cr
    #'   See the construction argument.
    #'   Use [`output_dim_for()`] to evaluate it.
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
    deep_clone = function(name, value) {
      if (name == "measure" && !is.null(value)) value$clone(deep = TRUE) else super$deep_clone(name, value)
    }
  )
)

#' @title Create a Generic Torch Task
#'
#' @description
#' Creates a [`TaskTorch`], the general-purpose task type of `mlr3torch`, from a `data.frame()` or a
#' [`DataBackend`][mlr3::DataBackend].
#' See [`TaskTorch`] for what you have to specify about the learning problem and what the
#' trade-offs are.
#'
#' @param x (`data.frame()` or [`DataBackend`][mlr3::DataBackend])\cr
#'   The data.
#' @param target (`character()`)\cr
#'   The names of the target columns. May be empty.
#' @param id (`character(1)`)\cr
#'   The id of the task.
#' @param ... (any)\cr
#'   Further arguments passed to [`TaskTorch`]`$new()`, such as `output_dim`, `prediction_encoder`
#'   or `measure`.
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
as_task_torch = function(x, target = character(0), id = deparse(substitute(x))[1L], ...) {
  TaskTorch$new(id = assert_string(id), backend = x, target = target, ...)
}

#' @export
task_check_col_roles.TaskTorch = function(task, new_roles, ...) { # nolint
  if (length(new_roles$target)) {
    return(NextMethod())
  }
  # Unlike other supervised tasks, a TaskTorch may have no target columns at all.
  # We therefore skip the method of TaskSupervised, which insists on at least one target, and run
  # the checks of the Task base class directly.
  task_check_col_roles_base(task, new_roles, ...)
}

task_check_col_roles_base = function(task, new_roles, ...) {
  fun = utils::getS3method("task_check_col_roles", "Task", envir = asNamespace("mlr3"))
  fun(task, new_roles, ...)
}

#' @export
output_dim_for.TaskTorch = function(x, ...) { # nolint
  if (is.null(x$output_dim)) {
    stopf("Task '%s' has no `output_dim`. Pass one to the task, or size the network's output yourself (e.g. `out_features` of `nn(\"head\")`).", x$id) # nolint
  }
  # evaluated rather than stored, because the number of output units follows from the target
  # columns and those can change after the task was constructed
  assert_int(x$output_dim(task = x), lower = 1L, coerce = TRUE)
}

#' @export
get_target_batchgetter.TaskTorch = function(task, ...) { # nolint
  stopf("Task '%s' does not define how its target becomes a tensor -- what `y` has to look like follows from the loss, so it is the learner that decides. Pass `target_batchgetter` to the learner (e.g. `lrn(\"torch.module\")`) or overwrite the method for your own `LearnerTorch` subclass.", task$id) # nolint
}

#' @export
encode_prediction.TaskTorch = function(task, network_output, predict_type, ...) { # nolint
  if (is.null(task$prediction_encoder)) {
    stopf("Task '%s' has no `prediction_encoder`, so there is no way to turn the network's output into a prediction. Pass one to the task, or overwrite the private `.encode_prediction()` method of the learner.", task$id) # nolint
  }
  # the raw network output is passed on unchanged, so a `prediction_encoder` is also how the output
  # of a network with more than one head is encoded
  task$prediction_encoder(task = task, predict_tensor = network_output, predict_type = predict_type)
}
