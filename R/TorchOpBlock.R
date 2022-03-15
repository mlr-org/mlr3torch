#' @export
TorchOpBlock = R6Class("TorchOpBlock",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "block", param_vals = list(), .block) {
      self$.block = assert_graph(.block)
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
    .build = function(input, param_vals, task) {
      architecture = self$.block$train(task)[[2]]
      layer_block = reduce_architecture(architecture, task, input)
      return(layer_block)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("block", TorchOpBlock)
