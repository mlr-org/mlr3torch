#' Note that we changed the default from 0L to 1L for start dim
#' @export
TorchOpFlatten = R6Class(
  inherit = TorchOp,
  public = list(
    initialize = function(id = "flatten", param_vals = list()) {
      param_set = ps(
        start_dim = p_int(default = 2L, lower = 1L, tags = "train"),
        end_dim = p_int(default = -1L, lower = 1L, tags = "train", special_vals = list(-1L))
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task, y) {
      layer = invoke(nn_flatten, .args = param_vals)
      return(layer)
    },
    .operator = "flatten"
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("flatten", TorchOpFlatten)
