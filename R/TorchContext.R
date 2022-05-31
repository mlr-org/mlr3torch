#' @description
#' Context for training a TorchModel
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
    #' @param measures (named `list()`)\cr
    #'   A list containing the sublists `"train"` and `"valid"` that contain the measures that
    #'   will be used during training and validation.
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
        private$learner$history = rhs
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
    },
    #' @field measures (`???`)\cr
    #'   The measures.
    measures = function(rhs) {
      assert_ro_binding(rhs)
      private$.measures
    }
  )
)
