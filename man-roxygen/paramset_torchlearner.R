#' @section Parameters:
#'
#' * `batch_size` :: (`integer(1)`)\cr
#'   The batch size.
#' * `epochs` :: `integer(1)`\cr
#'   The number of epochs.
#' * `device` :: `character(1)`\cr
#'   The device. One of `"auto"`, `"cpu"`, or `"cuda"` or other values defined in `mlr_reflections$torch$devices`.
#'   The value is initialized to `"auto"`, which will select `"cuda"` if possible and `"cpu"` otherwise.
#' * `measures_train` :: [`Measure`] or `list()` of [`Measure`]s.
#'   Measures to be evaluated during training.
#' * `measures_valid` :: [`Measure`] or `list()` of [`Measure`]s.
#'   Measures to be evaluated during validation.
#' * `num_threads` :: `integer(1)`\cr
#'   The number of threads for intraop pararallelization (if `device` is `"cpu"`).
#'   This value is initialized to 1.
# * `drop_last` :: `logical(1)`\cr
#    Whether to drop the last training batch in each epoch during training. Default is `FALSE`.
#    This does not impact validation.
#' * `shuffle` :: `logical(1)`\cr
#'   Whether to shuffle the instances in the dataset. This value is initialized to `TRUE`.
#' * `seed` :: `integer(1)` or `"random"`\cr
#'   The seed that is used during training and prediction.
#'   This value is initialized to `"random"`, which means that a random seed will be sampled at the beginning of the
#'   training phase. This seed (either set or randomly sampled) is available via `$model$seed` after training
#'   and used during prediction.
#'   Note that by setting the seed during the training phase this will mean that by default (i.e. when `seed` is
#'   `"random"`), clones of the learner will use a different seed.
#'
#' Additionally there are the parameters for the optimizer, the loss function and the callbacks.
#' They are prefixed with `"opt."`, `"loss."` and `"cb.<callback id>."` respectively.
