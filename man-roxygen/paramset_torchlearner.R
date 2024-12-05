#' @section Parameters:
#'
#' **General**:
#'
#' The parameters of the optimizer, loss and callbacks,
#' prefixed with `"opt."`, `"loss."` and `"cb.<callback id>."` respectively, as well as:
#'
#' * `epochs` :: `integer(1)`\cr
#'   The number of epochs.
#' * `device` :: `character(1)`\cr
#'   The device. One of `"auto"`, `"cpu"`, or `"cuda"` or other values defined in `mlr_reflections$torch$devices`.
#'   The value is initialized to `"auto"`, which will select `"cuda"` if possible, then try `"mps"` and otherwise
#'   fall back to `"cpu"`.
#' * `num_threads` :: `integer(1)`\cr
#'   The number of threads for intraop pararallelization (if `device` is `"cpu"`).
#'   This value is initialized to 1.
#' * `num_interop_threads` :: `integer(1)`\cr
#'   The number of threads for intraop and interop pararallelization (if `device` is `"cpu"`).
#'   This value is initialized to 1.
#'   Note that this can only be set once during a session and changing the value within an R session will raise a warning.
#' * `seed` :: `integer(1)` or `"random"` or `NULL`\cr
#'   The torch seed that is used during training and prediction.
#'   This value is initialized to `"random"`, which means that a random seed will be sampled at the beginning of the
#'   training phase. This seed (either set or randomly sampled) is available via `$model$seed` after training
#'   and used during prediction.
#'   Note that by setting the seed during the training phase this will mean that by default (i.e. when `seed` is
#'   `"random"`), clones of the learner will use a different seed.
#'   If set to `NULL`, no seeding will be done.
#' * `tensor_dataset` :: `logical(1)`\cr
#'   Whether to load all batches at once at the beginning of training and stack them.
#'   This is initialized to `FALSE`.
#'   When your dataset fits into memory this will make the loading of batches more efficient.
#'   When shuffle is `FALSE` (default), this means that each batch is constructed as a view of these tensors.
#'   Note that this should not be set for datasets that contain [`lazy_tensor`]s with random data augmentation,
#'   as this augmentation will only be applied once at the beginning of training.
#'
#' **Evaluation**:
#' * `measures_train` :: [`Measure`][mlr3::Measure] or `list()` of [`Measure`][mlr3::Measure]s.\cr
#'   Measures to be evaluated during training.
#' * `measures_valid` :: [`Measure`][mlr3::Measure] or `list()` of [`Measure`][mlr3::Measure]s.\cr
#'   Measures to be evaluated during validation.
#' * `eval_freq` :: `integer(1)`\cr
#'   How often the train / validation predictions are evaluated using `measures_train` / `measures_valid`.
#'   This is initialized to `1`.
#'   Note that the final model is always evaluated.
#'
#' **Early Stopping**:
#' * `patience` :: `integer(1)`\cr
#'   This activates early stopping using the validation scores.
#'   If the performance of a model does not improve for `patience` evaluation steps, training is ended.
#'   Note that the final model is stored in the learner, not the best model.
#'   This is initialized to `0`, which means no early stopping.
#'   The first entry from `measures_valid` is used as the metric.
#'   This also requires to specify the `$validate` field of the Learner, as well as `measures_valid`.
#' * `min_delta` :: `double(1)`\cr
#'   The minimum improvement threshold (`>`) for early stopping.
#'   Is initialized to 0.
#' 
'
#' **Dataloader**:
#' * `batch_size` :: `integer(1)`\cr
#'   The batch size (required).
#' * `shuffle` :: `logical(1)`\cr
#'   Whether to shuffle the instances in the dataset. Default is `FALSE`.
#'   This does not impact validation.
#' * `sampler` :: [`torch::sampler`]\cr
#'   Object that defines how the dataloader draw samples.
#' * `batch_sampler` :: [`torch::sampler`]\cr
#'   Object that defines how the dataloader draws batches.
#' * `num_workers` :: `integer(1)`\cr
#'   The number of workers for data loading (batches are loaded in parallel).
#'   The default is `0`, which means that data will be loaded in the main process.
#' * `collate_fn` :: `function`\cr
#'   How to merge a list of samples to form a batch.
#' * `pin_memory` :: `logical(1)`\cr
#'   Whether the dataloader copies tensors into CUDA pinned memory before returning them.
#' * `drop_last` :: `logical(1)`\cr
#'   Whether to drop the last training batch in each epoch during training. Default is `FALSE`.
#' * `timeout` :: `numeric(1)`\cr
#'   The timeout value for collecting a batch from workers.
#'   Negative values mean no timeout and the default is `-1`.
#' * `worker_init_fn` :: `function(id)`\cr
#'   A function that receives the worker id (in `[1, num_workers]`) and is exectued after seeding
#'   on the worker but before data loading.
#' * `worker_globals` :: `list()` | `character()`\cr
#'   When loading data in parallel, this allows to export globals to the workers.
#'   If this is a character vector, the objects in the global environment with those names
#'   are copied to the workers.
#' * `worker_packages` :: `character()`\cr
#'   Which packages to load on the workers.
#'
#' Also see `torch::dataloder` for more information.
