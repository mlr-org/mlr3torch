#' @title Layer Norm
#' @description
#' Normalizes over the last 'dims' dimensions of a tensor.
#' @section Parameters:
#' * `dims` :: `integer(1)`\cr
#'   The nunber of dimnensions over which will be normalized (starting from the last dimension).
#'
#' @export
TorchOpLayerNorm = R6Class("TorchOpLayerNorm",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "layer_norm", param_vals = list()) {
      param_set = ps(
        dims = p_int(lower = 1L, tags = c("train", "required")),
        elementwise_affine = p_lgl(default = TRUE, tags = c("required", "train"))
      )
      param_set$values = list(
        elementwise_affine = TRUE
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set)
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task) {
      input = inputs$input
      s = inputs$input$shape
      dims = param_vals$dims
      param_vals$dims = NULL

      assert_true(dims < length(s), .var.name = "parameter 'dims'")
      ld = tail(s, n = dims)
      param_vals = insert_named(param_vals, list(normalized_shape = ld))
      invoke(nn_layer_norm, .args = param_vals)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("layer_norm", TorchOpLayerNorm)
