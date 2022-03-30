#' TODO: This should not stay that way we can save the computation in the last attention layer
#' if we are only interested in the [CLS] Token.
#' @export
TorchOpIndexSelect = R6Class("TorchOpIndexSelect",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "indexselect", param_vals = list()) {
      param_set = ps(
        dim = p_int(default = 2L, lower = 0L),
        index = p_int(default = -1L)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "indexselect",
    .build = function(inputs, param_vals, task, y) {
      dim = param_vals[["dim"]] %??% 2L
      index = param_vals[["index"]] %??% -1L
      layer = nn_module(
        "indexselect",
        forward = function(input) {
          torch_index_select(input, dim, torch_tensor(index, dtype = torch_int()))
        }
      )()
      return(layer)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("indexselect", TorchOpIndexSelect)
