#' @title Measure for a Generic Torch Task
#'
#' @name mlr_measures_torch
#'
#' @description
#' Wraps a plain R function into a [`Measure`][mlr3::Measure] that scores the predictions of a
#' [`TaskTorch`].
#' Use [`msr_torch()`] to construct one.
#'
#' The function receives whichever of the arguments `truth`, `response`, `prob`, `se`, `prediction`,
#' `task`, `learner` and `train_set` it declares, and returns a single number.
#' What `truth`, `response` and `prob` look like depends on the task, see section *Inference* of
#' [`TaskTorch`].
#' Declaring a `task`, `learner` or `train_set` argument automatically adds the corresponding
#' `"requires_*"` property, which is what makes `mlr3` pass it.
#' A task without target columns has no `truth`, so its measures read the ground truth from the
#' `task` instead.
#'
#' @param id (`character(1)`)\cr
#'   The id of the measure.
#' @param fun (`function()`)\cr
#'   The scoring function, see above.
#' @param minimize (`logical(1)`)\cr
#'   Whether a smaller score is better.
#' @param range (`numeric(2)`)\cr
#'   The range of possible scores.
#' @param predict_type (`character(1)`)\cr
#'   The predict type the measure requires: `"response"` (default), `"prob"` or `"se"`.
#' @param properties (`character()`)\cr
#'   Properties of the measure, see [`Measure`][mlr3::Measure].
#' @param label (`character(1)`)\cr
#'   The label of the measure.
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
    initialize = function(id, fun, minimize = TRUE, range = c(-Inf, Inf), predict_type = "response",
      properties = character(), label = NA_character_) {
      private$.fun = assert_function(fun)
      args = names(formals(fun))
      # asking for one of these is what tells mlr3 to pass it to `$score()`
      for (arg in c("task", "learner", "train_set")) {
        if (arg %in% args) {
          properties = union(properties, paste0("requires_", arg))
        }
      }
      super$initialize(
        id = assert_string(id),
        task_type = "torch",
        predict_type = assert_choice(predict_type, pt_predict_types),
        properties = properties,
        range = assert_numeric(range, len = 2L, any.missing = FALSE),
        minimize = assert_flag(minimize),
        label = label,
        man = "mlr3torch::mlr_measures_torch"
      )
    }
  ),
  active = list(
    #' @field hash (`character(1)`)\cr
    #'   The hash of the measure.
    #'   [`Measure`][mlr3::Measure] hashes the private `$.score()` method, which is the same for
    #'   every `MeasureTorch`, so the scoring function is folded in here.
    #'   Without this, two measures with the same `id` but different functions would be
    #'   indistinguishable to everything that caches by hash.
    hash = function(rhs) {
      assert_ro_binding(rhs)
      calculate_hash(super$hash, hash_input(private$.fun))
    }
  ),
  private = list(
    .fun = NULL,
    .score = function(prediction, task = NULL, learner = NULL, train_set = NULL, ...) {
      args = list(truth = prediction$truth, response = prediction$response,
        prob = prediction$prob, se = prediction$se,
        task = task, learner = learner, train_set = train_set, prediction = prediction)
      args = args[intersect(names(formals(private$.fun)), names(args))]
      invoke(private$.fun, .args = args)
    }
  )
)

#' @title Create a Measure for a Generic Torch Task
#'
#' @description
#' Short form for constructing a [`MeasureTorch`], the quick way to score the predictions of a
#' [`TaskTorch`] with a plain R function.
#'
#' @param id (`character(1)`)\cr
#'   The id of the measure.
#' @param fun (`function()`)\cr
#'   The scoring function.
#'   It receives whichever of the arguments `truth`, `response`, `prob`, `se`, `prediction`, `task`,
#'   `learner` and `train_set` it declares, and returns a single number.
#' @param minimize (`logical(1)`)\cr
#'   Whether a smaller score is better.
#' @param range (`numeric(2)`)\cr
#'   The range of possible scores.
#' @param predict_type (`character(1)`)\cr
#'   The predict type the measure requires: `"response"` (default), `"prob"` or `"se"`.
#' @param properties (`character()`)\cr
#'   Properties of the measure, see [`Measure`][mlr3::Measure].
#'   The `"requires_task"`, `"requires_learner"` and `"requires_train_set"` properties are added
#'   automatically when `fun` declares the corresponding argument.
#' @param label (`character(1)`)\cr
#'   The label of the measure.
#' @return [`MeasureTorch`]
#' @export
#' @examplesIf torch::torch_is_installed()
#' # multi-label hamming loss, where truth and response are logical matrices
#' msr_torch("hamming", function(truth, response) mean(as.matrix(truth) != response))
#'
#' # a measure that reads the ground truth from the task, which is what a task without a target needs
#' msr_torch("reconstruction", function(task, prediction) {
#'   truth = as.matrix(task$data(rows = prediction$row_ids, cols = task$feature_names))
#'   mean((truth - prediction$response)^2)
#' }, range = c(0, Inf))
msr_torch = function(id, fun, minimize = TRUE, range = c(-Inf, Inf), predict_type = "response",
  properties = character(), label = NA_character_) {
  MeasureTorch$new(id = id, fun = fun, minimize = minimize, range = range,
    predict_type = predict_type, properties = properties, label = label)
}

#' @title Default Measure of a Generic Torch Task
#'
#' @name mlr_measures_torch.default
#'
#' @description
#' Delegates to the `$measure` field of the [`TaskTorch`] that is being scored.
#' This is the default measure of the task type `"torch"`, i.e. what `$score()` and `$aggregate()`
#' use when they are called without arguments.
#' It errors if the task does not carry a measure.
#'
#' @family Measure
#' @export
#' @examplesIf torch::torch_is_installed()
#' d = data.frame(x = rnorm(10), y = rnorm(10))
#' task = as_task_torch(d, target = "y",
#'   measure = msr_torch("mse", function(truth, response) mean((truth - response)^2)))
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
        minimize = TRUE,
        label = "Default Measure of the Task",
        man = "mlr3torch::mlr_measures_torch.default"
      )
    }
  ),
  private = list(
    .score = function(prediction, task, learner = NULL, train_set = NULL, ...) {
      measure = task$measure
      if (is.null(measure)) {
        stopf("Task '%s' has no default measure, pass a measure explicitly or construct the task with the `measure` argument.", task$id) # nolint
      }
      measure$score(prediction, task = task, learner = learner, train_set = train_set)
    }
  )
)
