#' @title Softmax
#'
#' @usage NULL
#' @name mlr_pipeops_torch_softmax
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_softmax description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("Softmax")`
#' `r roxy_param_id("softmax")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#'
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
#'
#' @section Internals:
#' Calls [`torch::nn_softmax()`] when trained.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_softmax")
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 7))
#'
#' # pot
#' obj = pot("softmax")
#' obj$id
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
