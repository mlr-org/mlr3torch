#' @title Context for Torch Learner
#'
#' @name mlr_context_torch
#'
#' @description
#' Context for training a torch learner.
#' This is the - mostly read-only - information callbacks have access to through the argument `ctx`.
#' For more information on callbacks, see [`CallbackSet`].
#'
#' @family Callback
#' @export
ContextTorch = R6Class("ContextTorch",
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @param learner ([`Learner`])\cr
    #'   The torch learner.
    #' @param task_train ([`Task`])\cr
    #'   The training task.
    #' @param task_valid ([`Task`] or `NULL`)\cr
    #'   The validation task.
    #' @param loader_train ([`torch::dataloader`])\cr
    #'   The data loader for training.
    #' @param loader_valid ([`torch::dataloader`] or `NULL`)\cr
    #'   The data loader for validation.
    #' @param measures_train (`list()` of [`Measure`]s or `NULL`)\cr
    #'   Measures used for training. Default is `NULL`.
    #' @param measures_valid (`list()` of [`Measure`]s or `NULL`)\cr
    #'   Measures used for validation.
    #' @param network ([`torch::nn_module`])\cr
    #'   The torch network.
    #' @param optimizer ([`torch::optimizer`])\cr
    #'   The optimizer.
    #' @param loss_fn ([`torch::nn_module`])\cr
    #'   The loss function.
    #' @param total_epochs (`integer(1)`)\cr
    #'   The total number of epochs the learner is trained for.
    #' @param prediction_encoder (`function()`)\cr
    #'   The learner's prediction encoder.
    initialize = function(learner, task_train, task_valid = NULL, loader_train, loader_valid = NULL,
      measures_train = NULL, measures_valid = NULL, network, optimizer, loss_fn, total_epochs, prediction_encoder) {
      self$learner = assert_r6(learner, "Learner")
      self$task_train = assert_r6(task_train, "Task")
      self$task_valid = assert_r6(task_valid, "Task", null.ok = TRUE)
      self$loader_train = assert_class(loader_train, "dataloader")
      self$loader_valid = assert_class(loader_valid, "dataloader", null.ok = TRUE)
      self$measures_train = assert_list(measures_train, names = "unique", any.missing = FALSE, types = "Measure",
        null.ok = TRUE) %??% list()
      self$measures_valid = assert_list(measures_valid, names = "unique", any.missing = FALSE, types = "Measure",
        null.ok = TRUE) %??% list()
      self$network = assert_class(network, "nn_module")
      self$optimizer = assert_class(optimizer, "torch_optimizer")
      self$loss_fn = assert_class(loss_fn, "nn_module")
      self$total_epochs = assert_integerish(total_epochs, lower = 0, any.missing = FALSE)
      self$last_scores_train = structure(list(), names = character(0))
      self$last_scores_valid = structure(list(), names = character(0))
      self$prediction_encoder = assert_function(prediction_encoder, args = c("predict_tensor", "task"))
      self$epoch = 0
      self$batch = 0
    },
    #' @field learner ([`Learner`])\cr
    #'   The torch learner.
    learner = NULL,
    #' @field task_train ([`Task`])\cr
    #'   The training task.
    task_train = NULL,
    #' @field task_valid ([`Task`] or `NULL`)\cr
    #'   The validation task.
    task_valid = NULL,
    #' @field loader_train ([`torch::dataloader`])\cr
    #'   The data loader for training.
    loader_train = NULL,
    #' @field loader_valid ([`torch::dataloader`])\cr
    #'   The data loader for validation.
    loader_valid = NULL,
    #' @field measures_train (`list()` of [`Measure`]s or `NULL`)\cr
    #'   Measures used for training.
    measures_train = NULL,
    #' @field measures_valid (`list()` of [`Measure`]s or `NULL`)\cr
    #'   Measures used for validation.
    measures_valid = NULL,
    #' @field network ([`torch::nn_module`])\cr
    #'   The torch network.
    network = NULL,
    #' @field optimizer ([`torch::optimizer`])\cr
    #'   The optimizer.
    optimizer = NULL,
    #' @field loss_fn ([`torch::nn_module`])\cr
    #'   The loss function.
    loss_fn = NULL,
    #' @field total_epochs (`integer(1)`)\cr
    #'   The total number of epochs the learner is trained for.
    total_epochs = NULL,
    #' @field last_scores_train (named `list()`)\cr
    #'   The scores from the last training batch. Names are the ids of the training measures.
    last_scores_train = NULL,
    #' @field last_scores_valid (`list()`)\cr
    #'   The scores from the last validation batch. Names are the ids of the validation measures.
    last_scores_valid = NULL,
    #' @field epoch (`integer(1)`)\cr
    #'   The current epoch.
    epoch = NULL,
    #' @field batch (`integer(1)`)\cr
    #'   The current iteration of the batch.
    batch = NULL,
    #' @field prediction_encoder (`function()`)\cr
    #'   The learner's prediction encoder.
    prediction_encoder = NULL
  )
)
