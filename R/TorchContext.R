#' @description
#' Context for training a TorchModel
#'
# @param learner ([`mlr3::Learner`][mlr3::Learner])\cr
#   The Torch Learner.
#' @export
ContextTorch = R6Class("ContextTorch",
  lock_objects = FALSE,
  public = list(
    #' @description Initializes an object of this [R6][R6::R6Class] class.
    initialize = function(learner, task, history, measures) {
      private$.learner = learner
      private$.task = task
      private$.history = history
      private$.measures = measures
    }
  ),
  private = list(
    .learner = NULL,
    .task = NULL,
    .history = NULL,
    .measures = NULL
  ),
  active = list(
    #' @field rhs (`nn_module()`)\cr
    #'   The torch network.
    network = function(rhs) {
      if (missing(rhs)) {
        private$learner$model$network
      } else {
        private$learner$model$network = rhs
      }
    },
    #' @field rhs (`nn_i()`)\cr
    #'   The torch network.
    optimizer = function(rhs) {
      if (missing(rhs)) {
        private$.model$optimizer
      } else {
        private$learner$model$optimizer
      }
    },
    loss_fn = function(rhs) {
      if (missing(rhs)) {
        private$.model$loss_fn
      } else {
        private$learner$loss_fn = rhs
      }
    },
    history = function(rhs) {
      if (missing(rhs)) {
        private$.history
      } else {
        private$learner$history = rhs
      }
    },
    task = function(rhs) {
      assert_ro_binding(rhs)
      private$.task
    },
    learner = function(rhs) {
      assert_ro_binding(rhs)
      private$.learner
    },
    measures = function(rhs) {
      assert_ro_binding(rhs)
      private$.measures
    }
  )
)
