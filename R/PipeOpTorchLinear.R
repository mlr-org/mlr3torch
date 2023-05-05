#' @title Linear Layer
#'
#' @usage NULL
#' @name pipeop_torch_linear
#' @format `r roxy_format(PipeOpTorchLinear)`
#'
#' @inherit torch::nnf_linear description
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchLinear)`
#' * `r roxy_param_id("nn_linear")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `out_features` :: `integer(1)`\cr
#'   The output features of the linear layer.
#' * `bias` :: `logical(1)`\cr
#'   Whether to use a bias.
#'   Default is `TRUE`.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`torch::nn_linear()`] when trained where the parameter `in_features` is inferred as the second
#' to last dimension of the input tensor.
#' @section Credit: `r roxy_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' obj = po("nn_linear", out_features = 10)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 7))
PipeOpTorchLinear = R6Class("PipeOpTorchLinear",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_linear", param_vals = list()) {
      param_set = ps(
        out_features = p_int(1L, Inf, tags = c("train", "required")),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_linear
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals, list(in_features = tail(shapes_in[[1]], 1)))
    },
    .shapes_out = function(shapes_in, param_vals, task) list(c(head(shapes_in[[1]], -1), param_vals$out_features))
  )
)

#' @include zzz.R
register_po("nn_linear", PipeOpTorchLinear)
