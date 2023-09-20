#' @title Indicate Loading of Torch Data
#' @description
#' @export
PipeOpTorchLoad = R6Class("PipeOpTorchLoad",
  inherit = PipeOp,
  public = list(
    initialize = function(id = "torch_load", param_vals = list()) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        input = data.table(name = "input", train = "ModelDescriptor", predict = "Task"),
        output = data.table(name = "output", train = "ModelDescriptor", predict = "Task"),
        tags = "torch"
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      inputs[[1]]$.cache_fence = unique(c(inputs[[1L]]$.cache_fence, inputs[[1L]]$.pointer[1L]))

    },
    .predict = function(inputs) {
      inputs
    }
  )
)

#' @export
register_po("torch_load", PipeOpTorchLoad)
