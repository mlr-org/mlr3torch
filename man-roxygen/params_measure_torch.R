#' @param id (`character(1)`)\cr
#'   The id of the measure.
#' @param fun (`function()`)\cr
#'   The scoring function.
#'   It receives whichever of the arguments `truth`, `response`, `prob`, `se`, `prediction`, `task`,
#'   `learner`, `train_set` and `weights` it declares, and must return a single number.
#'   An argument that the prediction does not have -- `weights` on a task without a
#'   `weights_measure` column, or `prob` on a response-only prediction -- is not passed at all, so a
#'   default declared for it is what the function sees.
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
#'   `"requires_model"` is not: a `learner` argument only says that the learner object is needed,
#'   and `mlr3` hands that over even when the model was not stored -- after a `resample()` with
#'   `store_models = FALSE`, `learner$network` is then `NULL` and a measure reading it scores
#'   whatever an empty model gives. Pass `properties = "requires_model"` yourself whenever the
#'   measure reaches for the trained network, so that `mlr3` refuses to score instead.
#' @param label (`character(1)`)\cr
#'   The label of the measure.
#' @param obs_loss (`function()` or `NULL`)\cr
#'   The per-observation loss. Declared like `fun` and if specified adds the `"obs_loss"` property.
#'   It must return one number per observation: a multi-target loss reduces over the targets
#'   (`rowMeans()`), not over the observations (`mean()`), and returning a single number is an error
#'   rather than a column of that number repeated.
