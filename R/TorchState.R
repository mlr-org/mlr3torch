#' @title Context where torch Callbacks are evaluted
#' @description
#' State for training a TorchModel.
#' This is the - mostly read-only - information callbacks have access to.
#'
#' @export
TorchState = R6Class("ContextTorch",
  lock_objects = FALSE,
  public = list(
    #' @description Initializes an object of this [R6][R6::R6Class] class.
    #' @param learner ([`mlr3::Learner`][mlr3::Learner])\cr
    #'   The torch learner.
    initialize = function(learner, task_train, task_valid = NULL, loader_train, loader_valid = NULL, measures_train = NULL, measures_valid = NULL, network, optimizer, loss_fn, total_epochs) {
      self$learner = assert_r6(learner, "Learner")
      self$task_train = assert_r6(task_train, "Task")
      self$task_valid = assert_r6(task_valid, "Task", null.ok = TRUE)
      self$loader_train = assert_class(loader_train, "dataloader")
      self$loader_valid = assert_class(loader_valid, "dataloader", null.ok = TRUE)
      self$measures_train = assert_list(measures_train, names = "unique", any.missing = FALSE, types = "Measure", null.ok = TRUE) %??% list()
      self$measures_valid = assert_list(measures_valid, names = "unique", any.missing = FALSE, types = "Measure", null.ok = TRUE) %??% list()
      self$network = assert_class(network, "nn_module")
      self$optimizer = assert_class(optimizer, "torch_optimizer")
      self$loss_fn = assert_class(loss_fn, "nn_module")
      self$total_epochs = assert_integerish(total_epochs, lower = 0, any.missing = FALSE)
      self$epoch = 0
      self$batch = 0
    },
    learner = NULL,
    task_train = NULL,
    task_valid = NULL,
    loader_train = NULL,
    loader_valid = NULL,
    train_iter = NULL,
    valid_iter = NULL,
    measures_train = NULL,
    measures_valid = NULL,
    total_epochs = NULL,
    last_scores_train = structure(list(), names = character(0)),
    last_scores_valid = structure(list(), names = character(0)),
    last_loss = NULL,
    epoch = NULL,
    batch = NULL
  )
)
