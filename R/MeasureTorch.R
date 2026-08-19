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
#' Passing `obs_loss` additionally gives the measure a per-observation loss, which is what
#' `$obs_loss()` and `as.data.table(prediction)` of a [`ResampleResult`][mlr3::ResampleResult]
#' report. It is declared the same way as `fun`, except that there is no `train_set` to ask for,
#' and it returns one number per observation rather than one number.
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
#' @param obs_loss (`function()` or `NULL`)\cr
#'   The per-observation loss, see above. If `NULL` (default), the measure has none and
#'   `$obs_loss()` returns `NA`, which is what [`Measure`][mlr3::Measure] does without the
#'   `"obs_loss"` property.
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
      properties = character(), label = NA_character_, obs_loss = NULL) {
      private$.fun = assert_function(fun)
      private$.obs_loss_fun = assert_function(obs_loss, null.ok = TRUE)
      if (!is.null(obs_loss)) {
        # `Measure$obs_loss()` is not given a `train_set`, so a function asking for one could never
        # be called -- rejected here rather than at the first scoring
        assert_subset(names(formals(obs_loss)), setdiff(mt_arg_names, "train_set"),
          .var.name = "arguments of `obs_loss`")
        properties = union(properties, "obs_loss")
      }
      args = c(names(formals(fun)), if (!is.null(obs_loss)) names(formals(obs_loss)))
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

# what a `MeasureTorch` function may ask for; `train_set` is only available when scoring
mt_arg_names = c("truth", "response", "prob", "se", "prediction", "task", "learner", "train_set")

mt_invoke = function(fun, prediction, task = NULL, learner = NULL, train_set = NULL) {
  args = list(truth = prediction$truth, response = prediction$response,
    prob = prediction$prob, se = prediction$se,
    task = task, learner = learner, train_set = train_set, prediction = prediction)
  args = args[intersect(names(formals(fun)), names(args))]
  invoke(fun, .args = args)
}

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
#' @param obs_loss (`function()` or `NULL`)\cr
#'   The per-observation loss. Declared like `fun`, except that there is no `train_set` to ask for,
#'   and returns one number per observation. Adds the `"obs_loss"` property.
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
msr_torch = function(id, fun, minimize = TRUE, range = c(-Inf, Inf), predict_type = "response",
  properties = character(), label = NA_character_, obs_loss = NULL) {
  MeasureTorch$new(id = id, fun = fun, minimize = minimize, range = range,
    predict_type = predict_type, properties = properties, label = label, obs_loss = obs_loss)
}

#' @title Default Measure of a Generic Torch Task
#'
#' @name mlr_measures_torch.default
#'
#' @description
#' Delegates to the `$default_measure` field of the [`TaskTorch`] that is being scored.
#' This is the default measure of the task type `"torch"`, i.e. what `$score()` and `$aggregate()`
#' use when they are called without arguments.
#' It errors if the task does not carry a measure.
#'
#' Whether a smaller score is better is a property of the task's measure, and this one is
#' constructed before any task is in sight, so `minimize` is `NA` -- unknown -- unless you say
#' otherwise. `mlr3` refuses to tune with an `NA` direction, which is the point: guessing it would
#' mean silently optimizing the wrong way. Pass `minimize` (and `range`) to tune against the
#' default measure of a task, and scoring a task whose measure disagrees is then an error.
#'
#' @param minimize (`logical(1)`)\cr
#'   Whether a smaller score is better, see above. Default is `NA`.
#' @param range (`numeric(2)`)\cr
#'   The range of possible scores. Defaults to the unbounded range, for the same reason.
#'
#' @family Measure
#' @export
#' @examplesIf torch::torch_is_installed()
#' d = data.frame(x = rnorm(10), y = rnorm(10))
#' task = as_task_torch(d, target = "y",
#'   default_measure = msr_torch("mse", function(truth, response) mean((truth - response)^2)))
#' msr("torch.default")
#'
#' # to tune against it, state the direction of the task's measure
#' msr("torch.default", minimize = TRUE)
MeasureTorchDefault = R6Class("MeasureTorchDefault",
  inherit = Measure,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(minimize = NA, range = c(-Inf, Inf)) {
      super$initialize(
        id = "torch.default",
        task_type = "torch",
        predict_type = "response",
        properties = "requires_task",
        range = assert_numeric(range, len = 2L, any.missing = FALSE),
        minimize = assert_flag(minimize, na.ok = TRUE),
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
      # A tuner reads `minimize` long before it scores anything, so a disagreement here means it
      # has been ranking the archive in the wrong direction.
      if (!is.na(self$minimize) && !isTRUE(self$minimize == measure$minimize)) {
        stopf("Measure 'torch.default' was constructed with minimize = %s, but the default measure '%s' of task '%s' has minimize = %s.", self$minimize, measure$id, task$id, measure$minimize) # nolint
      }
      measure$score(prediction, task = task, learner = learner, train_set = train_set)
    }
  )
)
