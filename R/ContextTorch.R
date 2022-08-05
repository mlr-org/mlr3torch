#' @title Context where torch Callbacks are evaluted
#' @description
#' Context for training a TorchModel.
#' This is the - mostly read-only - information callbacks have access to.
#'
#' @export
ContextTorch = R6Class("ContextTorch",
  lock_objects = FALSE,
  public = list(
    #' @description Initializes an object of this [R6][R6::R6Class] class.
    #' @param learner ([`mlr3::Learner`][mlr3::Learner])\cr
    #'   The torch learner.
    #' @param task ([`mlr3::Task`][mlr3::Task])\cr
    #'   The machine learning task.
    #' @param history ([`History`][History])\cr
    #'   The history for the torch learner.
    initialize = function(learner, task, history, train_loader, valid_loader, epoch = NULL) {
      private$.learner = learner
      private$.task = task
      private$.history = history
      private$.train_loader = train_loader
      private$.valid_loader = valid_loader
      if (is.null(epoch)) {
        self$epoch = 0L
      } else {
        self$epoch = epoch
      }
    },
    epoch = NULL
  ),
  private = list(
    .learner = NULL,
    .task = NULL,
    .history = NULL,
    .train_iter = NULL,
    .valid_iter = NULL,
    .epoch = NULL
  ),
  active = list(
    #' @field train_loader (`dataloader`)\cr
    #'   The training loader.
    train_loader = function(rhs) {
      assert_ro_binding(rhs)
      private$.train_loader
    },
    #' @field valid_loader (`dataloader`)\cr
    #'   The validation loader.
    valid_loader = function(rhs) {
      assert_ro_binding(rhs)
      private$.valid_loader
    },
    #' @field y_hat (`list(1)`)\cr
    #'   The previous prediction.
    y_hat = function(rhs) {
      assert_ro_binding(rhs)
      private$.y_hat
    },
    #' @field y (`list(1)`)\cr
    #'   The previous truth.
    y = function(rhs) {
      assert_ro_binding(rhs)
      private$.y
    },
    #' @field pred_train (`list(1)`)\cr
    #'   The predictions of the validation phase.
    pred_train = function(rhs) {
      assert_ro_binding(rhs)
      private$.pred_train
    },
    #' @field pred_valid (`list(1)`)\cr
    #'   The predictions of the validation phase.
    pred_valid = function(rhs) {
      assert_ro_binding(rhs)
      private$.pred_valid
    },
    #' @field train_iter (`integer(1)`)\cr
    #'   The training iteration.
    train_iter = function(rhs) {
      assert_ro_binding(rhs)
      private$.train_iter
    },
    #' @field valid_iter (`integer(1)`)\cr
    #'   The validation iteration.
    valid_iter = function(rhs) {
      assert_ro_binding(rhs)
      private$.valid_iter
    },
    #' @field network (`torch::nn_module()`)\cr
    #'   The torch network.
    network = function(rhs) {
      if (missing(rhs)) {
        private$learner$model$network
      } else {
        private$learner$model$network = rhs
      }
    },
    #' @field optimizer (`torch::optimizer()`)\cr
    #'   The torch optimizer
    optimizer = function(rhs) {
      if (missing(rhs)) {
        private$.model$optimizer
      } else {
        private$learner$model$optimizer
      }
    },
    #' @field loss_fn (`torch::nn_module()`)\cr
    #'   The torch loss function.
    loss_fn = function(rhs) {
      if (missing(rhs)) {
        private$.model$loss_fn
      } else {
        private$learner$loss_fn = rhs
      }
    },
    #' @field history ([`History`][History])\cr
    #'   The learning history.
    history = function(rhs) {
      if (missing(rhs)) {
        private$.history
      } else {
        private$.learner$history = rhs
      }
    },
    #' @field task ([`Task`][mlr3::Task])\cr
    #'   The machine learning task.
    task = function(rhs) {
      assert_ro_binding(rhs)
      private$.task
    },
    #' @field learner ([`Learner`][mlr3::Learner])\cr
    #'   The torch learner.
    learner = function(rhs) {
      assert_ro_binding(rhs)
      private$.learner
    }
  )
)
