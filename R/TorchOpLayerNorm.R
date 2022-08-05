#' @title Layer Norm
#' @description
#' Normalizes over the last 'n_dim' dimensions of a tensor.
#' @section Parameters:
#' * `dims` :: `integer(1)`\cr
#'   The nunber of dimnensions over which will be normalized (starting from the last dimension).
#'
#' @template param_id
#' @template param_param_vals
#'
#' @export
TorchOpLayerNorm = R6Class("TorchOpLayerNorm",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "layer_norm", param_vals = list()) {
      param_set = ps(
        n_dim = p_int(lower = 1L, tags = c("train", "required")),
        elementwise_affine = p_lgl(default = TRUE, tags = c("required", "train")),
        eps = p_dbl(default = 1e-5, lower = 0, tags = "train")
      )
      param_set$values = list(
        elementwise_affine = TRUE
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set)
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      input = inputs$input
      s = inputs$input$shape
      n_dim = param_vals$n_dim
      param_vals$n_dim = NULL
      ld = tail(s, n = n_dim)
      args = insert_named(param_vals, list(normalized_shape = ld))
      invoke(nn_layer_norm, .args = args)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("layer_norm", TorchOpLayerNorm)
