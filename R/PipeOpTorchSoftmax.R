#' @title Softmax
#'
#' @name mlr_pipeops_torch_softmax
#'
#' @inherit torch::nnf_softmax description
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
#'
#' @section Internals:
#' Calls [`torch::nn_softmax()`] when trained.
#' @family PipeOps
#' @export
#' @examples
#' obj = po("nn_softmax", dim = 2)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 7))
PipeOpTorchSoftmax = R6::R6Class("PipeOpTorchSoftmax",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
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
