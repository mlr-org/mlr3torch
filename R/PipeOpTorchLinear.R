#' @title Linear Layer
#' @inherit torch::nnf_linear description
#' @section nn_module:
#' Calls [`torch::nn_linear()`] when trained where the parameter `in_features` is inferred as the second
#' to last dimension of the input tensor.
#' @section Parameters:
#' * `out_features` :: `integer(1)`\cr
#'   The output features of the linear layer.
#' * `bias` :: `logical(1)`\cr
#'   Whether to use a bias.
#'   Default is `TRUE`.
#'
#' @templateVar id nn_linear
#' @template pipeop_torch_channels_default
#' @templateVar param_vals out_features = 10
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchLinear = R6Class("PipeOpTorchLinear",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_linear", param_vals = list()) {
      param_set = ps(
        out_features = p_int(1L, Inf, tags = c("train", "required")),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_linear,
        only_batch_unknown = FALSE
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      d_in = tail(shapes_in[[1]], 1)
      if (is.na(d_in)) {
        stopf("PipeOpLinear received an input shape where the last dimension is unknown. Please provide an input with a known last dimension.")
      }
      c(param_vals, list(in_features = d_in))
    },
    .shapes_out = function(shapes_in, param_vals, task) list(c(head(shapes_in[[1]], -1), param_vals$out_features))
  )
)

#' @include aaa.R
register_po("nn_linear", PipeOpTorchLinear)
