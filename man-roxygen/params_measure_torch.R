#' @param id (`character(1)`)\cr
#'   The id of the measure.
#' @param fun (`function()`)\cr
#'   The scoring function.
#'   It receives whichever of the arguments `truth`, `response`, `prob`, `se`, `prediction`, `task`,
#'   `learner`, `train_set` and `weights` it declares, and must return a single number.R
#' @param minimize (`logical(1)`)\cr
#'   Whether a smaller score is better.
#'   `NA` (default) means the direction is unknown.
#' @param range (`numeric(2)`)\cr
#'   The range of possible scores.
#' @param predict_type (`character(1)`)\cr
#'   The predict type the measure requires: `"response"` (default), `"prob"` or `"se"`.
#' @param properties (`character()`)\cr
#'   Properties of the measure, see [`Measure`][mlr3::Measure].
#'   The `"requires_task"`, `"requires_learner"`, `"requires_train_set"` and `"weights"` properties
#'   are added automatically when `fun` declares the corresponding argument.
#' @param label (`character(1)`)\cr
#'   The label of the measure.
#' @param obs_loss (`function()` or `NULL`)\cr
#'   The per-observation loss. Declared like `fun` and if specified adds the `"obs_loss"` property.
