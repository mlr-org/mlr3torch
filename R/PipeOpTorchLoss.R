#' @title Loss Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_losstorch loss
#' @description
#' Configures the `loss` of a [`ModelDescriptor`].
#'
#' @section Construction:
#' ```
#' PipeOpTorchLoss$new(loss, id = "torch_loss", param_vals = list())
#' ```
#' `r roxy_param_id("torch_loss")`
#' `r roxy_param_param_vals()`
#' * `loss` :: [`TorchLoss`].
#'   The loss function.
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' The parameters of the constructor argument `loss`.
#'
#' @section Internals:
#' TODO:
#'
#' @section Fields:
#' Only fields inherited from [`PipeOp`].
#'
#' @section Methods:
#' Only ,methods inherited from [`PipeOp`].
#'
#' @seealso PipeOpTorch, PipeOpTorchOptimizer
#' @export
#' @examples
#' g = top("input") %>>%
#'   top("select", items = "num") %>>%
#'   top("output") %>>%
#'   top("loss", "mse")
#'
#' task = tsk("iris")
#' g$train(task)
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
