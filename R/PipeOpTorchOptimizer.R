#' @title PipeOp Optimizer
#'
#' @usage NULL
#' @name mlr_pipeops_torch_optimizer
#' @format `r roxy_pipeop_torch_format()`
#'
#' @description
#' Configures the optimizer of a deep learning model.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchOptimizer)`
#' * `optimizer` :: [`TorchOptimizer`]\cr
#'   The [optimizer][TorchOptimizer].
#' * `r roxy_param_id("torch_optimizer")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' The `ParamSet` is set to the `ParamSet` of the provided optimizer.
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: See the respective child class.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' po_opt = po("torch_optimizer", optimizer = t_opt("sgd"), lr = 0.01)
#' po_opt$param_set
#' md = (po("torch_ingress_num") %>>% po("nn_head"))$train(tsk("iris"))
#' md[[1L]]$optimizer
#' md = po_opt$train(md)
#' md[[1L]]$optimizer
PipeOpTorchOptimizer = R6Class("PipeOpTorchOptimizer",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(optimizer, id = "torch_optimizer", param_vals = list()) {
      private$.optimizer = assert_r6(as_torch_optimizer(optimizer), "TorchOptimizer")
      super$initialize(
        id = id,
        param_set = alist(private$.optimizer$param_set),
        param_vals = param_vals,
        module_generator = NULL,
        inname = "input"
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      expect_true(is.null(inputs[[1L]]$optimizer))
      inputs[[1]]$optimizer = private$.optimizer$clone(deep = TRUE)
      inputs
    },
    .optimizer = NULL
  )
)

#' @include zzz.R
register_po("torch_optimizer", PipeOpTorchOptimizer)
