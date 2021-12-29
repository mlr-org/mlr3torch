#' @description NeuralNetwork
#' This
NeuralNetwork = R6Class("NeuralNetwork",
  inherit = mlr3::Learner,
  public = list(
    #' List of [nn_module]s
    layers = NULL,

    initialize = function(architecture, task) {
      self$layers = build_model(task, architecture)
    },
    forward = function(x) {
      for (layer in self$layers) {
        x = self$layer(x)
      }
    },
    append = function(layer) {
      assert(inherits(layer, "nn_module"))
      self$layers = c(self$layers, layer)
    }
  ),
  active = list(
    length = function() {
      return(length(self$layers))
    }
  )
)
