#' @title TorchOpSoftmax
#' @include TorchOpSoftmax.R
#' @export
TorchOpSoftmax = R6::R6Class("TorchOpSoftmax",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "softmax", param_vals = list()) {
      param_set = ps(
        dim = p_int(1L, Inf, tags = c("train", "required"))
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "linear",
    .build = function(input, param_vals, task, y) {
      layer = invoke(nn_softmax, .args = param_vals)
      return(layer)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("softmax", value = TorchOpSoftmax)
