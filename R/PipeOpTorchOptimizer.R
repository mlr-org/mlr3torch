#' @title Optimizer Configuration
#'
#' @usage NULL
#' @name mlr_pipeops_torch_optimizer
#' @format `r roxy_format(PipeOpTorchOptimizer)`
#'
#' @description
#' Configures the optimizer of a deep learning model.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchOptimizer)`
#' * `optimizer` :: [`TorchOptimizer`] or `character(1)` or `torch_optimizer_generator`\cr
#'   The optimizer (or something convertible via [`as_torch_optimizer()`]).
#'   This object is cloned during construction.
#' * `r roxy_param_id("torch_optimizer")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels:
#' There is one input channel `"input"` and one output channel `"output"`.
#' During *training*, the channels are of class [`ModelDescriptor`].
#' During *prediction*, the channels are of class [`Task`].
#'
#' @section State:
#' The state is set to an empty `list()`.
#'
#' @section Parameters:
#' The parameters are defined dynamically from the optimizer that is set during construction.
#' @section Fields:
#' Only fields inherited from [`PipeOp`].
#' @section Methods:
#' Only methods inherited from [`PipeOp`].
#' @section Internals:
#' During training, the optimizer is cloned and added to the [`ModelDescriptor`].
#' Note that the parameter set of the stored [`TorchOptimizer`] is reference-identical to the parameter set of the
#' pipeop itself.
#' @family model_configuration
#' @export
#' @examples
#' po_opt = po("torch_optimizer", "sgd", lr = 0.01)
#' po_opt$param_set
#' mdin = po("torch_ingress_num")$train(list(tsk("iris")))
#' mdin[[1L]]$optimizer
#' mdout = po_opt$train(mdin)
#' mdout[[1L]]$optimizer
PipeOpTorchOptimizer = R6Class("PipeOpTorchOptimizer",
  inherit = PipeOp,
  public = list(
    initialize = function(optimizer = t_opt("adam"), id = "torch_optimizer", param_vals = list()) {
      private$.optimizer = as_torch_optimizer(optimizer, clone = TRUE)
      input = data.table(name = "input", train = "ModelDescriptor", predict = "Task")
      output = data.table(name = "output", train = "ModelDescriptor", predict = "Task")
      super$initialize(
        id = id,
        param_set = alist(private$.optimizer$param_set),
        param_vals = param_vals,
        input = input,
        output = output,
        packages = private$.optimizer$packages
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      if (!test_null(inputs[[1L]]$optimizer)) {
        stopf("The optimizer of the model descriptor is already configured.")
      }
      inputs[[1]]$optimizer = private$.optimizer$clone(deep = TRUE)
      self$state = list()
      inputs
    },
    .predict = function(inputs) inputs,
    .optimizer = NULL
  )
)

#' @include zzz.R
register_po("torch_optimizer", PipeOpTorchOptimizer)
