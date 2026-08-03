#' @title Dropout
#' @inherit torch::nnf_dropout description
#' @section nn_module:
#' Calls [`torch::nn_dropout()`] when trained.
#' @section Parameters:
#' * `p` :: `numeric(1)`\cr
#'  Probability of an element to be zeroed. Default: 0.5.
#' * `inplace` :: `logical(1)`\cr
#'   If set to `TRUE`, will do this operation in-place. Default: `FALSE`.
#'
#' @templateVar id nn_dropout
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchDropout = R6Class("PipeOpTorchDropout",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
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

#' @include aaa.R
register_po("nn_dropout", PipeOpTorchDropout)
