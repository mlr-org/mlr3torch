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
#'   Note that this can only be set **once** per session, so setting this for one learner also changes the
#'   behavior of other learners, and a later learner asking for a different value errors.
#'   `NULL` (default) uses whatever is set.
#'   In order to use different values for this parameter, use encapsulation to train the learners
#'   in separate R sessions.
#' * `seed` :: `integer(1)` or `"random"` or `NULL`\cr
#'   The torch seed that is used during training and prediction.
#'   This value is initialized to `"random"`, which means that a random seed will be sampled at the beginning of the
#'   training phase. This seed (either set or randomly sampled) is available via `$model$seed` after training
#'   and used during prediction.
#'   Note that by setting the seed during the training phase this will mean that by default (i.e. when `seed` is
#'   `"random"`), clones of the learner will use a different seed.
#'   If set to `NULL`, no seeding will be done.
#'   This parameter only seeds torch's random number generator, it does **not** seed R's.
#'   Anything that is drawn from R's RNG is therefore unaffected by it, so to make those parts
#'   reproducible you need to seed R's RNG as well, e.g. via [`set.seed()`].
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
#' **Resuming**:
#' * `path` :: `character(1)` or `TRUE`\cr
#'   Continues training from a checkpoint written by [`t_clbk("checkpoint")`][mlr_callback_set.checkpoint],
#'   also in a new R session.
#'   Either the path of the folder that the checkpoint callback wrote to -- the most recent
#'   checkpoint in it is used -- or `TRUE`, which takes the path from the checkpoint callback of
#'   this learner.
#'   The network, the optimizer and the states of the callbacks are restored, and training then
#'   continues until `epochs` is reached.
#'   Note that `epochs` is the *total* number of epochs, i.e. it includes the epochs the checkpoint
#'   was already trained for: resuming a checkpoint from epoch 5 with `epochs = 8` trains 3 more
#'   epochs.
#'   If the folder contains no checkpoint, training starts from scratch, so that the same script can
#'   be used to start a run and to continue it after an interruption.
#'   This is unset by default, i.e. no resuming.
#'
#'   The learning rate schedule, the training history and the other callback states are restored,
#'   but the state of the random number generator is not, so a resumed run does not reproduce the
#'   uninterrupted run it continues.
#'
#'   A `path` is one run's folder, so it must not be shared by the iterations of a `resample()` or
#'   `benchmark()`: every iteration after the first would resume the first one's checkpoint, find it
#'   already at `epochs` and train nothing.
#'   Use [`t_clbk("checkpoint")`][mlr_callback_set.checkpoint] with a `path` function, which is
#'   called once per run, and set this parameter to `TRUE` to follow it.
#'
#'   Callback states are matched to the callbacks of the resuming run **by id and nothing else**.
#'   A state whose id is not among this run's callbacks is skipped with a warning, and a callback
#'   whose id is not in the checkpoint simply starts fresh -- adding a callback to a resumed run is
#'   allowed and is not reported.
#'   Giving two structurally different callbacks the same id in the two runs is therefore not
#'   detected: the state of one is fed into the other, which can restore a nonsensical state without
#'   any error or warning.
#'   Resume with the same callback ids the checkpoint was written with, or with different ones.
#'   What a callback restores is documented in the *Resuming* section of its own help page.
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
#'   When a run is resumed (see *Resuming*), the best score, the epoch it was observed in and the
#'   number of evaluation steps without improvement are restored, so `patience` keeps counting
#'   across runs instead of starting over.
#'   The best epoch's weights are not part of a checkpoint, however -- they are a full copy of the
#'   network, which every checkpoint would otherwise carry.
#'   A resumed run with `restore_best_weights` that never beats the restored best score therefore
#'   ends with the weights of its last epoch while `$internal_tuned_values` still reports the
#'   earlier best epoch, which is warned about when the state is restored.
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
