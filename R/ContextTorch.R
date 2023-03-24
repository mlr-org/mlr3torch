#' @title Context for Torch Learner
#'
#' @usage NULL
#' @name mlr_context_torch
#' @format `r roxy_format(ContextTorch)`
#'
#' @description
#' Context for training a torch learner.
#' This is the - mostly read-only - information callbacks have access to through the argument `ctx`.
#' For more information on callbacks, see [`CallbackTorch`].
#'
#' @section Construction:
#' `r roxy_construction(ContextTorch)`
#' * `learner` :: [`Learner`]\cr
#'   The torch learner.
#' * `task_train` :: [`Task`]\cr
#'   The training task.
#' * `task_valid` :: [`Task`]\cr
#'   The validation task. Default is `NULL`.
#' * `loader_train` :: [`torch::dataloader`]
#'   The data loader for training.
#' * `loader_valid` :: [`torch::dataloader`]
#'   The data loader for validation.
#' * `measures_train` :: `list()` of [`Measure`]s\cr
#'   Measures used for training. Default is `NULL`.
#' * `measures_valid` :: `list()` of [`Measure`]s\cr
#'   Measures used for validation. Default is `NULL`.
#' * `network` :: [`torch::nn_module`]\cr
#'   The torch network.
#' * `optimizer` :: [`torch::optimizer`]\cr
#'   The optimizer.
#' * `loss_fn` :: [`torch::nn_module`]\cr
#'   The loss function.
#' * `total_epochs` :: `integer(1)`\cr
#'   The total number of epochs the learner is trained for.

#' @section Fields:
#' All objects configured in the initialization, as well as:
#' * `last_scores_train` :: named `list()`\cr
#'   The scores from the last training batch. Names are the ids of the training measures.
#' * `last_scores_valid` :: named `list()`\cr
#'   The scores from the last validation batch. Names are the ids of the validation measures.
#' * `epoch` :: `integer(1)`\cr
#'   The current epoch.
#'   The current epoch.
#' * `batch` :: `list()` of [`torch::torch_tensor`]\cr
#'   The current iteration of the batch.
#'
#' @section Methods:
#' There are no methods.
#'
#' @family callback
#' @export
ContextTorch = R6Class("ContextTorch",
  inherit = Context,
  lock_objects = FALSE,
  public = list(
    initialize = function(learner, task_train, task_valid = NULL, loader_train, loader_valid = NULL,
      measures_train = NULL, measures_valid = NULL, network, optimizer, loss_fn, total_epochs) {
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
      self$epoch = 0
      self$batch = 0
    },
    learner = NULL,
    task_train = NULL,
    task_valid = NULL,
    loader_train = NULL,
    loader_valid = NULL,
    measures_train = NULL,
    measures_valid = NULL,
    network = NULL,
    optimizer = NULL,
    loss_fn = NULL,
    total_epochs = NULL,
    last_scores_train = NULL,
    last_scores_valid = NULL,
    epoch = NULL,
    batch = NULL
  )
)
