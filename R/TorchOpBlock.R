#' @title Create a TorchOpBlock
#' @export
TorchOpBlock = R6Class("TorchOpBlock",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "block", param_vals = list(), .block) {
      private$.block = assert_graph(.block)
      super$initialize(
        id = id,
        param_set = .block$param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "block",
    .block = NULL,
    .build = function(inputs, param_vals, task, y) {
      architecture = private$.block$train(task)[[2]]
      layer_block = architecture_reduce(architecture, task, input)
      return(layer_block)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("block", TorchOpBlock)
