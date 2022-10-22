#' @title Softmax
#'
#' @usage NULL
#' @name pipeop_torch_softmax
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_softmax description
#'
#' @section Module:
#' Calls [`torch::nn_softmax()`] when trained.
#'
#' @template pipeop_torch_channels_default
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
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
PipeOpTorchSoftmax = R6::R6Class("PipeOpTorchSoftmax",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_softmax", param_vals = list()) {
      param_set = ps(
        dim = p_int(1L, Inf, tags = c("train", "required"))
      )
      super$initialize(
        id = id,
        module_generator = nn_softmax,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  )
)

#' @include zzz.R
register_po("nn_softmax", PipeOpTorchSoftmax)
