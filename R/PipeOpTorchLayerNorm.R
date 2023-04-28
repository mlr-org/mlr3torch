#' @title Layer Normalization
#'
#' @usage NULL
#' @name mlr_pipeops_torch_layer_norm
#' @format `r roxy_format(PipeOpTorchLayerNorm)`
#'
#' @inherit torch::nnf_layer_norm description
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchLayerNorm)`
#' * `r roxy_param_id("nn_layer_norm")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#' @section Parameters:
#' * `dims` :: `integer(1)`\cr
#'   The number of dimensions over which will be normalized (starting from the last dimension).
#' * `elementwise_affine` :: `logical(1)`\cr
#'   Whether to learn affine-linear parameters initialized to `1` for weights and to `0` for biases.
#'   The default is `TRUE`.
#' * `eps` :: `numeric(1)`\cr
#'   A value added to the denominator for numerical stability.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`torch::nn_layer_norm()`] when trained.
#' The parameter `normalized_shape` is inferre as the dimensions of the last `dims` dimensions of the input shape.
#' @section Credit: `r roxy_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' obj = po("nn_layer_norm", dims = 2)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 7))
PipeOpTorchLayerNorm = R6Class("PipeOpTorchLayerNorm",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_layer_norm", param_vals = list()) {
      param_set = ps(
        dims = p_int(lower = 1L, tags = c("train", "required")),
        elementwise_affine = p_lgl(default = TRUE, tags = "train"),
        eps = p_dbl(default = 1e-5, lower = 0, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_layer_norm)
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      assert_int(param_vals$dims, upper = length(shapes_in))
      param_vals$normalized_shape = utils::tail(shapes_in[[1L]], param_vals$dims)
      param_vals$dims = NULL
      param_vals
    }
  )
)

#' @include zzz.R
register_po("nn_layer_norm", PipeOpTorchLayerNorm)
