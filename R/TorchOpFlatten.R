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
    .build = function(input, param_vals, task) {
      start_dim = param_vals[["start_dim"]] %??% 2L
      end_dim = param_vals[["end_dim"]] %??% -1L
      layer = nn_flatten(start_dim, end_dim)
      return(layer)
    },
    .operator = "flatten"
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("flatten", TorchOpFlatten)

nn_flatten = nn_module("nn_flatten",
  initialize = function(start_dim, end_dim) {
    self$start_dim = start_dim
    self$end_dim = end_dim
  },
  forward = function(input) {
    torch_flatten(input, start_dim = self$start_dim, end_dim = self$end_dim)
  }
)
