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
#'   The number of threads for intraop parallelization (if `device` is `"cpu"`).
#'   This value is initialized to 1.
#'   When resampling, benchmarking or tuning in parallel, each worker uses this many threads, so
#'   divide the available cores among the workers instead of setting this to the number of cores.
#' * `num_interop_threads` :: `integer(1)`\cr
#'   The number of threads for interop parallelization (if `device` is `"cpu"`).
#' * `seed` :: `integer(1)` or `"random"` or `NULL`\cr
#'   The torch seed that is used during training and prediction.
#'   This value is initialized to `"random"`, which means that a random seed will be sampled at the beginning of the
#'   training phase. This seed (either set or randomly sampled) is available via `$model$seed` after training
#'   and used during prediction.
#'   Note that by setting the seed during the training phase this will mean that by default (i.e. when `seed` is
#'   `"random"`), clones of the learner will use a different seed.
#'   If set to `NULL`, no seeding will be done.
#' * `tensor_dataset` :: `logical(1)` | `"device"`\cr
#'   Whether to load all batches at once at the beginning of training and stack them.
#'   This is initialized to `FALSE`.
#'   If set to `"device"`, the device of the tensors will be set to the value of `device`, which
#'   can avoid unnecessary moving of tensors between devices.
#'   When your dataset fits into memory this will make the loading of batches faster.
#'   Note that this should not be set for datasets that contain [`lazy_tensor`]s with random data augmentation,
#'   as this augmentation will only be applied once at the beginning of training.
#'
#' **Evaluation**:
#' * `measures_train` :: [`Measure`][mlr3::Measure] or `list()` of [`Measure`][mlr3::Measure]s\cr
#'   Measures to be evaluated during training.
#' * `measures_valid` :: [`Measure`][mlr3::Measure] or `list()` of [`Measure`][mlr3::Measure]s\cr
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
#'   Note that this counts *evaluation steps*, not epochs: when `eval_freq` is greater than `1`,
#'   `patience` evaluation steps correspond to `patience * eval_freq` epochs.
#'   Note that the final model is stored in the learner, not the best model.
#'   This is initialized to `0`, which means no early stopping.
#'   The first entry from `measures_valid` is used as the metric.
#'   This also requires to specify the `$validate` field of the Learner, as well as `measures_valid`.
#'   If this is set, the epoch after which no improvement was observed, can be accessed via the `$internal_tuned_values`
#'   field of the learner.
#' * `min_delta` :: `double(1)`\cr
#'   The minimum improvement threshold for early stopping.
#'   Is initialized to 0.
#' * `restore_best_weights` :: `logical(1)`\cr
#'   Whether to restore the weights of the best epoch when training ends, instead of keeping those
#'   of the last epoch that was trained. Is initialized to `FALSE`, i.e. the network of the last
#'   epoch is stored. Setting this to `TRUE` makes the stored network the one of the epoch that
#'   `$internal_tuned_values` reports, and costs one additional copy of the network's parameters in
#'   memory. Checkpoints written by `t_clbk("checkpoint")` are unaffected: they always hold the
#'   network as training left it.
#'
#' **Dataloader**:
#' * `batch_size` :: `integer(1)`\cr
#'   The batch size used by the training and prediction dataloader.
#'   It is required for training (unless a `batch_sampler` is provided, which already determines the
#'   batches) and it is required for prediction (unless `batch_size_predict` is set).
#' * `batch_size_predict` :: `integer(1)`\cr
#'   The batch size used by the prediction dataloader (this includes the validation data during
#'   training). When set, it overrides `batch_size` for prediction.
#'   The batch size does not change the predictions, but smaller batches take longer and require less
#'   memory.
#' * `shuffle` :: `logical(1)`\cr
#'   Whether to shuffle the instances in the dataset. This is initialized to `TRUE`,
#'   which differs from the default (`FALSE`).
#'   It is ignored when a `sampler` or `batch_sampler` is provided.
#' * `sampler` :: [`torch::sampler`]\cr
#'   Object that defines how the dataloader draws samples, i.e. the order in which the observations are
#'   drawn. This must be the sampler *generator* (as returned by [`torch::sampler()`]), not an instance,
#'   as it is instantiated with the training dataset internally.
#' * `batch_sampler` :: [`torch::sampler`]\cr
#'   Object that defines how the dataloader draws batches. As for `sampler`, this must be the generator.
#'   When it is provided, the parameters `batch_size`, `shuffle` and `drop_last` are ignored during
#'   training, because the batch sampler already determines the batches.
#' * `num_workers` :: `integer(1)`\cr
#'   The number of workers for data loading (batches are loaded in parallel).
#'   The default is `0`, which means that data will be loaded in the main process.
#' * `collate_fn` :: `function`\cr
#'   How to merge a list of samples to form a batch.
#' * `pin_memory` :: `logical(1)`\cr
#'   Whether the dataloader copies tensors into CUDA pinned memory before returning them.
#' * `drop_last` :: `logical(1)`\cr
#'   Whether to drop the last training batch in each epoch during training. Default is `FALSE`.
#'   It is ignored when a `batch_sampler` is provided.
#' * `timeout` :: `numeric(1)`\cr
#'   The timeout value for collecting a batch from workers.
#'   Negative values mean no timeout and the default is `-1`.
#' * `worker_init_fn` :: `function(id)`\cr
#'   A function that receives the worker id (in `[1, num_workers]`) and is executed after seeding
#'   on the worker but before data loading.
#' * `worker_globals` :: `list()` | `character()`\cr
#'   When loading data in parallel, this allows to export globals to the workers.
#'   If this is a character vector, the objects in the global environment with those names
#'   are copied to the workers.
#' * `worker_packages` :: `character()`\cr
#'   Which packages to load on the workers.
#'
#' Also see `torch::dataloader` for more information.
