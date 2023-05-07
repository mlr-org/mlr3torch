#' @title Layer Normalization
#'
#' @templateVar id nn_layer_norm
#' @templateVar param_vals dims = 1
#' @template pipeop_torch
#' @template pipeop_torch_channels_default
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_layer_norm description
#' @section Parameters:
#' * `dims` :: `integer(1)`\cr
#'   The number of dimensions over which will be normalized (starting from the last dimension).
#' * `elementwise_affine` :: `logical(1)`\cr
#'   Whether to learn affine-linear parameters initialized to `1` for weights and to `0` for biases.
#'   The default is `TRUE`.
#' * `eps` :: `numeric(1)`\cr
#'   A value added to the denominator for numerical stability.
#' @section Internals:
#' Calls [`torch::nn_layer_norm()`] when trained.
#' The parameter `normalized_shape` is inferred as the dimensions of the last `dims` dimensions of the input shape.
#' @export
PipeOpTorchLayerNorm = R6Class("PipeOpTorchLayerNorm",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_layer_norm", param_vals = list()) {
      param_set = ps(
        dims               = p_int(lower = 1L, tags = c("train", "required")),
        elementwise_affine = p_lgl(default = TRUE, tags = "train"),
        eps                = p_dbl(default = 1e-5, lower = 0, tags = "train")
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
