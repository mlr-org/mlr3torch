#' @title Torch Graph Learner
#' @description
#' Torch Graph Learner.
#' @export
GraphLearnerTorch = R6Class("GraphLearnerTorch",
  inherit = GraphLearner,
  public = list(
    initialize = function(graph, id = NULL, param_vals = list(), predict_type = NULL, clone_graph = TRUE) {
      private$.model_id = graph$output$op.id
      op_model = graph$pipeops[[private$.model_id]]
      cls = class(op_model)[1L]
      task_type = switch(cls,
        TorchOpModelClassif = "classif",
        TorchOpModelRegr = "regr",
        stopf("Unknown object of class '%s'.", class(op_model)[[1L]])
      )

      super$initialize(
        graph = graph,
        id = id,
        param_vals = param_vals,
        task_type = task_type,
        predict_type = predict_type,
        clone_graph = clone_graph
      )
    }
  ),
  active = list(
    #' @field parameters (`list()`)\cr
    #'   A list with the network's parameters.
    parameters = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$network$parameters
    },
    #' @field history ([`History][History])\cr
    #'   History of the training proceess.
    history = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$model[[private$.model_id]]$history
    },
    #' @field optimizer ([`torch_Optimizer`][torch::optimizer])\cr
    #'  The optimizer.
    optimizer = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$model[[private$.model_id]]$optimizer
    },
    #' @field loss_fn (`nn_loss()`)\cr
    #'   The loss function.
    loss_fn = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$model[[private$.model_id]]$loss_fn
    },
    #' @field network ([`nn_module()`][torch::nn_module])\cr
    #'   The network.
    network = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$model[[private$.model_id]]$network
    }
  ),
  private = list(
    .model_id = NULL
  )
)
