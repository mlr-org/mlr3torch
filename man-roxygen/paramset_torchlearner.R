#' @section Parameters:
#' * `batch_size` :: (`integer(1)`)\cr
#'   The batch size.
#' * `epochs` :: `integer(1)`\cr
#'   The number of epochs.
#' * `device` :: `character(1)`\cr
#'   The device. One of `"auto"`, `"cpu"`, or `"cuda"`.
#' * `measures_train` :: `list()` of [`Measure`]s.
#'   Measures to be evaluated during training.
#' * `measures_valid` :: `list()` of [`Measure`]s.
#'   Measures to be evaluated during validation.
#' * `augmentation` :: ??
#'  TODO:
#' * `callbacks` :: (list of) `CallbackTorch`\cr
#'   The callbacks to .
#' * `drop_last` :: `logical(1)`\cr
#'   Whether to drop the last batch in each epoch during training. Default is `FALSE`.
#' * `num_threads` :: `integer(1)`\cr
#'   The number of threads (if `device` is `"cpu"`). Default is 1.
#' * `shuffle` :: `logical(1)`\cr
#'   Whether to shuffle the instances in the dataset. Default is `TRUE`.
