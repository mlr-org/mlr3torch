#' @section Parameters:
#'
#' * `batch_size` :: (`integer(1)`)\cr
#'   The batch size.
#' * `epochs` :: `integer(1)`\cr
#'   The number of epochs.
#' * `device` :: `character(1)`\cr
#'   The device. One of `"auto"`, `"cpu"`, or `"cuda"`.
#' * `measures_train` :: [`Measure`] or `list()` of [`Measure`]s.
#'   Measures to be evaluated during training.
#' * `measures_valid` :: [`Measure`] or `list()` of [`Measure`]s.
#'   Measures to be evaluated during validation.
#' * `drop_last` :: `logical(1)`\cr
#'   Whether to drop the last batch in each epoch during training. Default is `FALSE`.
#' * `num_threads` :: `integer(1)`\cr
#'   The number of threads (if `device` is `"cpu"`). Default is 1.
#' * `shuffle` :: `logical(1)`\cr
#'   Whether to shuffle the instances in the dataset. Default is `TRUE`.
#' * `early_stopping_rounds` :: `integer(1)`\cr
#'   How many rounds to wait for early stopping. The default is 0.
#' * `seed` :: `integer(1)`\cr
#'   The seed that is used during training. The value `seed + 1` is used during prediction.
#'   If this is missing (default), a random seed is generated.
#'
#' Additionally there are the parameters for the optimizer, the loss function and the callbacks.
#' They are prefixed with `"opt."`, `"loss."` and `"cb.<callback id>"` respectively.
