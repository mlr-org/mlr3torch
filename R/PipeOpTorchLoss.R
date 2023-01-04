#' @title PipeOp Loss
#'
#' @usage NULL
#' @name mlr_pipeops_torch_loss
#' @format `r roxy_pipeop_torch_format()`
#'
#' @description
#' Configures the loss of a deep learning model.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchLoss)`
#' * `optimizer` :: [`TorchLoss`]\cr
#'   The [loss][TorchLoss].
#' * `r roxy_param_id("torch_loss")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' The `ParamSet` is set to the `ParamSet` of the provided loss.
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
PipeOpTorchLoss = R6Class("PipeOpTorchLoss",
  inherit = PipeOp,
  public = list(
    initialize = function(loss, id = "torch_loss", param_vals = list()) {
      private$.loss = assert_r6(as_torch_loss(loss), "TorchLoss")
      input = data.table(name = "input", train = "ModelDescriptor", predict = "Task")
      output = data.table(name = "output", train = "ModelDescriptor", predict = "Task")
      super$initialize(
        id = id,
        param_set = alist(private$.loss$param_set),
        param_vals = param_vals,
        input = input,
        output = output,
        packages = loss$packages
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      inputs[[1]]$loss = private$.loss$clone(deep = TRUE)
      inputs
    },
    .loss = NULL
  )
)

#' @include zzz.R
register_po("torch_loss", PipeOpTorchLoss)
