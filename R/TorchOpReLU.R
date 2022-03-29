#' @title TorchOpReLU
#' @include TorchOpReLU.R
#' @export
TorchOpReLU = R6Class("TorchOpReLU",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "relu", param_vals = list()) {
      param_set = ps(
        inplace = p_lgl(default = FALSE)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "relu",
    .build = function(input, param_vals, task, y) {
      invoke(nn_relu, .args = param_vals)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("relu", value = TorchOpReLU)
