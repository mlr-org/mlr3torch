#' @title Layer Norm
#'
#' @usage NULL
#' @template pipeop_torch_format
#'
#' @inherit torch::nn_layer_norm description
#'
#' @section Torch Module:
#' Wraps the torch module [`nn_layer_norm`][torch::nn_layer_norm].
#'
#' @template pipeop_torch_channels
#' @template pipeop_torch_state
#'
#' @section Parameters:
#' See arguments of [`torch::nn_layer_norm`].
#' * `dims` :: `integer(1)`\cr The numer of dimensions over which will be normalized (starting from the last dimension).
#' * `eps` :: `numeric(1)`\cr
#'   A value added to the denominator for numerical stability.
#'   Default: 1e-5.
#' * `elementwise_affine`\cr
#'   A boolean value that when set to ‘TRUE’, this module has learnable per-element affine parameters initialized to
#'   ones (for weights) and zeros (for biases). Default: ‘TRUE’.
#'
#' @section Internals:
#' The parameter `normalized_shape` of the wrapped `torch::nn_layer_norm` is inferred as the dimensions of the the
#' last `n_dim` layers.
#'
#' @template torch_license_docu
#'
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchLayerNorm = R6Class("PipeOpTorchLayerNorm",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
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
