#' @title Dropout
#'
#' @usage NULL
#' @name mlr_pipeops_torch_dropout
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_dropout description
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchDropout)`
#'
#' * `r roxy_param_id("nn_layer_norm")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `p` :: `numeric(1)`\cr
#'  Probability of an element to be zeroed. Default: 0.5 inplace
#' * `inplace` :: `logical(1)`\cr
#'   If set to `TRUE`, will do this operation in-place. Default: `FALSE.`
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`torch::nn_dropout()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' obj = po("nn_dropout")
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 7))
PipeOpTorchDropout = R6Class("PipeOpTorchDropout",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_dropout", param_vals = list()) {
      param_set = ps(
        p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_dropout
      )
    }
  )
)

#' @include zzz.R
register_po("nn_dropout", PipeOpTorchDropout)
