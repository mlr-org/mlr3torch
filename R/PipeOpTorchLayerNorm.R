#' @title Layer Normalization
#'
#' @usage NULL
#' @name pipeop_torch_layer_norm
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_layer_norm description
#'
#' @section Module:
#' Calls [`torch::nn_layer_norm()`] when trained.
#'
#' @template pipeop_torch_channels_default
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `dims` :: `integer(1)`\cr
#'   The number of dimensions over which will be normalized (starting from the last dimension).
#' * `elementwise_affine` :: `logical(1)`\cr
#'   Whether to learn affine-linear parameters initialized to `1` for weights and to `0` for biases.
#'   The default is `TRUE`.
#' * `eps` :: `numeric(1)`\cr
#'   A value added to the denominator for numerical stability.
#'   The default is `1e-5`.
#'
#' @examples
#' # po
#' obj = po("nn_layer_norm", n_dim = 2)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 7))
#'
#' # pot
#' obj = pot("layer_norm", n_dim = 2)
#' obj$id
#'
#' @template torch_license_docu
#'
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#'
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
