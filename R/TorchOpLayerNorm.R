#' @title Layer Norm
#' @description
#' layer normalization
#' @export
TorchOpLayerNorm = R6Class("TorchOpLayerNorm",
  inherit = TorchOp,
  public = list(
    initialize = function(id, param_vals) {
      param_set = ps(
        normalized_shape = p_uty(tags = c("required", "train")),
        elementwise_affine = p_lgl(default = TRUE, tags = c("required", "train"))
      )
      param_set$values = list(
        elementwise_affine = TRUE
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set)
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task, y) {
      invoke(nn_layer_norm, .args = param_vals)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("layer_norm", TorchOpLayerNorm)
