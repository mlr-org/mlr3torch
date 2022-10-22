#' @title Multi Layer Perceptron
#'
#' @name mlr_learners_classif.mlp
#'
#' @description
#' Simple multi layer perceptron with dropout.
#'
#' @section Architecture:
#' One layer is: $dropout(activation(linear(x)))$.
#' These layers are stacked and the final output layer is a simple classification layer.
#'
#' @section Parameters:
#' *Learner specific parameters*:
#' * `act` :: `character(1)`\cr
#'   Activation function.
#' * `act_args` :: named `list()`\cr
#'   A named list with initialization arguments for the activation function.
#' * `layers` :: `integer(1)`\cr
#'   The number of layers.
#' * `p` :: `numeric(1)`\cr
#'   The dropout probability.
#'
#' @templateVar id classif.mlp
#' @template learner
#' @template example
#' @export
LearnerClassifMLP = R6Class("LearnerClassifMLP",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template param_optimizer
    #' @template param_loss
    initialize = function(optimizer = "adam", loss = "cross_entropy") {
      param_set = ps(
        activation = p_fct(default = "relu", levels = torch_reflections$activation, tags = c("train", "network", "required")),
        activation_args = p_uty(default = list(), tags = c("train", "network")),
        layers = p_int(lower = 1L, tags = c("train", "network", "required")),
        d_hidden = p_int(lower = 1L, tags = c("train", "network", "required")),
        p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = c("train", "network", "required"))
      )
      param_set$values = list(activation = "relu")
      super$initialize(
        id = "classif.mlp",
        properties = c("weights", "twoclass", "multiclass", "hotstart_forward"),
        label = "Multi Layer Perceptron",
        param_set = param_set,
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners_classif.mlp",
        feature_types = c("numeric", "integer")
      )
    }
  ),
  private = list(
    .network = function(task) {
      pv = self$param_set$get_values(tags = "network")
      block = top("linear", out_features = pv$d_hidden) %>>%
        top("activation", fn = pv$activation, args = pv$activation_args) %>>%
        top("dropout", p = pv$p)

      graph = top("input") %>>%
        top("select", items = "num") %>>%
        top("repeat", block, times = pv$layer) %>>%
        top("output")

      network = graph$train(task)[[1L]]$network

      return(network)
    }
  )
)
