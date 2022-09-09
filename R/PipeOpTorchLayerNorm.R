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
PipeOpTorchLayerNorm = R6Class("PipeOpTorchLayerNorm",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_layer_norm", param_vals = list()) {
      param_set = ps(
        n_dim = p_int(lower = 1L, tags = c("train", "required")),
        elementwise_affine = p_lgl(default = TRUE, tags = "train"),
        eps = p_dbl(default = 1e-5, lower = 0, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_layer_norm)
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals) {
      assert_int(param_vals$n_dim, upper = length(shapes_in))
      param_vals$normalized_shape = utils::tail(shapes_in, param_vals$n_dim)
      param_vals$n_dim = NULL
      param_vals
    }
  )
)

#' @include zzz.R
register_po("nn_layer_norm", PipeOpTorchLayerNorm)
