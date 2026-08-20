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
#' @template params_measure_torch
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
      private$.obs_loss_fun = assert_function(obs_loss, null.ok = TRUE)
      if (!is.null(obs_loss)) {
        assert_subset(names(formals(obs_loss)),
          c("truth", "response", "prob", "se", "prediction", "task", "learner", "weights"),
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
    .score = function(prediction, task = NULL, learner = NULL, train_set = NULL, ...) {
      mt_invoke(private$.fun, prediction, task = task, learner = learner, train_set = train_set)
    },
    .obs_loss = function(prediction, task = NULL, learner = NULL, ...) {
      mt_invoke(private$.obs_loss_fun, prediction, task = task, learner = learner)
    }
  )
)

mt_invoke = function(fun, prediction, task = NULL, learner = NULL, train_set = NULL) {
  args = list(truth = prediction$truth, response = prediction$response,
    prob = prediction$prob, se = prediction$se,
    task = task, learner = learner, train_set = train_set, prediction = prediction,
    weights = prediction$weights)
  args = args[intersect(names(formals(fun)), names(args))]
  invoke(fun, .args = args)
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
        properties = "requires_task",
        range = c(-Inf, Inf),
        minimize = NA,
        label = "Default Measure for a TaskTorch",
        man = "mlr3torch::mlr_measures_torch.default"
      )
    }
  ),
  private = list(
    .score = function(prediction, task, learner = NULL, train_set = NULL, ...) {
      measure = task$default_measure
      if (is.null(measure)) {
        stopf("Task '%s' has no default measure, pass a measure explicitly or construct the task with the `default_measure` argument.", task$id) # nolint
      }
      measure$score(prediction, task = task, learner = learner, train_set = train_set)
    },
    .obs_loss = function(prediction, task, learner = NULL, ...) {
      stopf("Measure 'torch.default' does not delegate the per-observation loss, because whether the default measure of task '%s' has one is not known when this measure is constructed. Score with that measure directly.", task$id) # nolint
    }
  )
)
