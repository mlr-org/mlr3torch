#' @title Measure for a Generic Torch Task
#'
#' @name mlr_measures_torch
#'
#' @description
#' Wraps a plain R function into a [`Measure`][mlr3::Measure] that scores the predictions of a
#' [`TaskTorch`].
#' Use [`msr_torch()`] to construct one.
#' See the *Custom Learning Problems* article for how to create and use such measures.
#'
#' @param id (`character(1)`)\cr
#'   The id of the measure.
#' @param fun (`function()`)\cr
#'   The scoring function.
#'   It receives whichever of the arguments `truth`, `response`, `prob`, `se`, `lazy_tensor`,
#'   `prediction`, `task`, `learner`, `train_set` and `weights` it declares, and must return a
#'   single number. Asking for anything else is an error.
#' @param minimize (`logical(1)`)\cr
#'   Whether a smaller score is better.
#'   `NA` (default) means the direction is unknown.
#' @param range (`numeric(2)`)\cr
#'   The range of possible scores.
#' @param predict_type (`character(1)`)\cr
#'   The predict type the measure requires: `"response"` (default), `"prob"`, `"se"` or
#'   `"lazy_tensor"`.
#'   A measure asking for one it did not declare here still receives it, if the prediction has it.
#' @param properties (`character()`)\cr
#'   Properties of the measure, see [`Measure`][mlr3::Measure].
#'   The `"requires_task"`, `"requires_learner"`, `"requires_train_set"` and `"weights"` properties
#'   are added automatically when `fun` declares the corresponding argument.
#' @param label (`character(1)`)\cr
#'   The label of the measure.
#' @param obs_loss (`function()` or `NULL`)\cr
#'   The per-observation loss. Declared like `fun`, except that `train_set` is not available here,
#'   and if specified adds the `"obs_loss"` property.
#'   It must return one number per observation.
#'
#' @family Measure
#' @export
#' @examplesIf torch::torch_is_installed()
#' d = data.frame(x = rnorm(10), y = rnorm(10))
#' task = as_task_torch(d, target = "y")
#' measure = msr_torch("mse", function(truth, response) mean((truth - response)^2))
#' measure
MeasureTorch = R6Class("MeasureTorch",
  inherit = Measure,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, fun, minimize = NA, range = c(-Inf, Inf), predict_type = "response",
      properties = character(), label = NA_character_, obs_loss = NULL) {
      private$.fun = assert_function(fun)
      assert_subset(names(formals(fun)), mt_args, .var.name = "arguments of `fun`")
      private$.obs_loss_fun = assert_function(obs_loss, null.ok = TRUE)
      if (!is.null(obs_loss)) {
        # `Measure$obs_loss()` is not given a `train_set`, so a function asking for one could never
        # be called -- rejected here rather than at the first scoring
        assert_subset(names(formals(obs_loss)), setdiff(mt_args, "train_set"),
          .var.name = "arguments of `obs_loss`")
        properties = union(properties, "obs_loss")
      }
      args = c(names(formals(fun)), if (!is.null(obs_loss)) names(formals(obs_loss)))
      for (arg in c("task", "learner", "train_set")) {
        if (arg %in% args) {
          properties = union(properties, paste0("requires_", arg))
        }
      }
      if ("weights" %in% args) {
        properties = union(properties, "weights")
      }
      super$initialize(
        id = assert_string(id),
        task_type = "torch",
        predict_type = assert_choice(predict_type, pt_predict_types),
        properties = properties,
        range = assert_numeric(range, len = 2L, any.missing = FALSE),
        minimize = assert_flag(minimize, na.ok = TRUE),
        label = label,
        man = "mlr3torch::mlr_measures_torch"
      )
    }
  ),
  active = list(
    #' @field hash (`character(1)`)\cr
    #'   The hash of the measure.
    hash = function(rhs) {
      assert_ro_binding(rhs)
      calculate_hash(super$hash, private$.fun, private$.obs_loss_fun)
    }
  ),
  private = list(
    .fun = NULL,
    .obs_loss_fun = NULL,
    .score = function(prediction, task = NULL, learner = NULL, train_set = NULL, weights = NULL,
      ...) {
      # `mlr3` passes the weights of the prediction, or `NULL` if `$use_weights` says to ignore
      # them, so they are taken from here rather than from the prediction
      mt_invoke(private$.fun, prediction, self$id, task = task, learner = learner,
        train_set = train_set, weights = weights)
    },
    .obs_loss = function(prediction, task = NULL, learner = NULL, ...) {
      # `Measure$obs_loss()` passes no weights, so `$use_weights` is honoured here instead
      loss = mt_invoke(private$.obs_loss_fun, prediction, self$id, task = task, learner = learner,
        weights = if (self$use_weights == "use") prediction$weights)
      assert_numeric(loss, .var.name = sprintf("value of the `obs_loss` of measure '%s'", self$id))
      # `mlr3` assigns this with `data.table::set()`, which recycles: a function that reduced when
      # it should not have -- `mean()` where `rowMeans()` was meant -- would otherwise fill the
      # column with one number that looks like a per-observation loss
      if (length(loss) != length(prediction$row_ids)) {
        stopf("The `obs_loss` of measure '%s' returned %i values for %i observations. It has to return one per observation, so a multi-target loss reduces over the targets and not over the observations.", self$id, length(loss), length(prediction$row_ids)) # nolint
      }
      loss
    }
  )
)

# what a `MeasureTorch` function may ask for; `train_set` is only available when scoring, so
# `$.obs_loss()` leaves it out
mt_args = c("truth", "response", "prob", "se", "lazy_tensor", "prediction", "task", "learner",
  "train_set", "weights")

mt_invoke = function(fun, prediction, id, task = NULL, learner = NULL, train_set = NULL,
  weights = NULL) {
  args = list(truth = prediction$truth, response = prediction$response,
    prob = prediction$prob, se = prediction$se, lazy_tensor = prediction$lazy_tensor,
    task = task, learner = learner, train_set = train_set, prediction = prediction,
    weights = weights)
  args = args[intersect(names(formals(fun)), names(args))]
  required = required_args(fun)
  # A function asking for `weights` without a default cannot be called without them, and passing
  # `NULL` is what makes that silent: `sum(err * NULL) / n` is 0, a perfect score for a loss.
  if (is.null(weights) && "weights" %chin% required) {
    stopf("Measure '%s' asks for `weights` without a default, but there are none: the task has no `weights_measure` column, or the measure's `$use_weights` is \"ignore\". Give the argument a default, such as `weights = rep(1, length(truth))`, to also score without them.", id) # nolint
  }
  # An argument that is not there -- no `weights_measure` column, no `prob` -- is `NULL` here, and
  # passing it explicitly would override a default the function gave it. Drop those, so that
  # `function(truth, response, weights = 1)` sees its own default rather than `NULL`.
  invoke(fun, .args = args[!(map_lgl(args, is.null) & names(args) %nin% required)])
}

#' @title Create a Measure for a Generic Torch Task
#'
#' @description
#' Short form for constructing a [`MeasureTorch`].
#' See the *Custom Learning Problems* article for how to create and use such measures.
#'
#' @template params_measure_torch
#' @return [`MeasureTorch`]
#' @export
#' @examplesIf torch::torch_is_installed()
#' m = msr_torch("hamming", function(truth, response) mean(as.matrix(truth) != response))
#' m$properties
#'
#' # with a per-observation loss
#' m = msr_torch("mse", function(truth, response) mean((truth - response)^2),
#'   obs_loss = function(truth, response) (truth - response)^2)
#' m$properties
msr_torch = function(id, fun, minimize = NA, range = c(-Inf, Inf), predict_type = "response",
  properties = character(), label = NA_character_, obs_loss = NULL) {
  MeasureTorch$new(id = id, fun = fun, minimize = minimize, range = range,
    predict_type = predict_type, properties = properties, label = label, obs_loss = obs_loss)
}

#' @title Default Measure of a Generic Torch Task
#'
#' @name mlr_measures_torch.default
#'
#' @description
#' This is a simple placeholder measure and extracts the actual value from the `$default_measure`
#' of a [`TaskTorch`].
#'
#' @family Measure
#' @export
#' @examplesIf torch::torch_is_installed()
#' d = data.frame(x = rnorm(10), y = rnorm(10))
#' task = as_task_torch(d, target = "y",
#'   default_measure = msr_torch("mse", function(truth, response) mean((truth - response)^2)))
#' msr("torch.default")
MeasureTorchDefault = R6Class("MeasureTorchDefault",
  inherit = Measure,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      super$initialize(
        id = "torch.default",
        task_type = "torch",
        predict_type = "response",
        # the per-observation loss is delegated like the score is, so whether one exists depends on
        # the task's measure rather than on this one
        properties = c("requires_task", "obs_loss"),
        range = c(-Inf, Inf),
        minimize = NA,
        label = "Default Measure for a TaskTorch",
        man = "mlr3torch::mlr_measures_torch.default"
      )
    }
  ),
  private = list(
    .score = function(prediction, task, learner = NULL, train_set = NULL, ...) {
      measure = mtd_measure(task)
      measure$score(prediction, task = task, learner = learner, train_set = train_set)
    },
    .obs_loss = function(prediction, task, learner = NULL, ...) {
      # `Measure$obs_loss()` reports `NA` for a measure that has no per-observation loss, so a task
      # whose default measure has none behaves like any other such measure
      mtd_measure(task)$obs_loss(prediction, task = task, learner = learner)
    }
  )
)

mtd_measure = function(task) {
  measure = task$default_measure
  if (is.null(measure)) {
    stopf("Task '%s' has no default measure, pass a measure explicitly or construct the task with the `default_measure` argument.", task$id) # nolint
  }
  measure
}
