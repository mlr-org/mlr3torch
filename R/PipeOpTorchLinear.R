#' @title Linear Layer
#'
#' @usage NULL
#' @name pipeop_torch_linear
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_linear description
#'
#' @section Module:
#' Calls [`torch::nn_linear()`] when trained.
#'
#' @template pipeop_torch_channels_default
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `out_features` :: `integer(1)`\cr
#'   The output features of the linear layer.
#' * `bias` :: `logical(1)`\cr
#'   Whether to use a bias.
#'   Default is `TRUE`.
#'
#' @examples
#' # po
#' obj = po("nn_linear", out_features = 10)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 7))
#'
#' # pot
#' obj = pot("linear", out_features = 10)
#' obj$id
#'
#' @template torch_license_docu
#'
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#'
#' @export
PipeOpTorchLinear = R6Class("PipeOpTorchLinear",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
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
    .shape_dependent_params = function(shapes_in, param_vals) {
      c(param_vals, list(in_features = tail(shapes_in[[1]], 1)))
    },
    .shapes_out = function(shapes_in, param_vals) list(c(head(shapes_in[[1]], -1), param_vals$out_features))
  )
)

#' @include zzz.R
register_po("nn_linear", PipeOpTorchLinear)

