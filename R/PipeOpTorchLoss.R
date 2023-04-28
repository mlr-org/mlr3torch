#' @title Loss Configuration
#'
#' @usage NULL
#' @name mlr_pipeops_torch_loss
#' @format `r roxy_format(PipeOpTorchLoss)`
#'
#' @description
#' Configures the loss of a deep learning model.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchLoss)`
#' * `loss` :: [`TorchLoss`] or `character(1)` or `nn_loss`\cr
#'   The loss (or something convertible via [`as_torch_loss()`]).
#'   This object is cloned during construction.
#' * `r roxy_param_id("torch_loss")`
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
#' The parameters are defined dynamically from the loss set during construction.
#' @section Fields:
#' Only fields inherited from [`PipeOp`].
#' @section Methods:
#' Only methods inherited from [`PipeOp`].
#' @section Internals:
#' During training the loss is cloned and added to the [`ModelDescriptor`].
#' @family model_configuration
#' @export
#' @examples
#' po_loss = po("torch_loss", "cross_entropy")
#' po_loss$param_set
#' mdin = po("torch_ingress_num")$train(list(tsk("iris")))
#' mdin[[1L]]$loss
#' mdout = po_loss$train(mdin)[[1L]]
#' mdout$loss
PipeOpTorchLoss = R6Class("PipeOpTorchLoss",
  inherit = PipeOp,
  public = list(
    initialize = function(loss, id = "torch_loss", param_vals = list()) {
      private$.loss = as_torch_loss(loss, clone = TRUE)
      input = data.table(name = "input", train = "ModelDescriptor", predict = "Task")
      output = data.table(name = "output", train = "ModelDescriptor", predict = "Task")
      super$initialize(
        id = id,
        param_set = alist(private$.loss$param_set),
        param_vals = param_vals,
        input = input,
        output = output,
        packages = private$.loss$packages
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      if (!test_null(inputs[[1L]]$loss)) {
        stopf("The loss of the model descriptor is already configured.")
      }
      assert_true(is.null(inputs[[1L]]$loss))
      inputs[[1]]$loss = private$.loss$clone(deep = TRUE)
      self$state = list()
      inputs
    },
    .loss = NULL
  )
)

#' @include zzz.R TorchLoss.R
register_po("torch_loss", PipeOpTorchLoss)
