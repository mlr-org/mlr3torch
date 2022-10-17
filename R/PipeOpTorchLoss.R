#' @title Loss Function
#'
#' @usage NULL
#' @description
#' Configures the `loss` of a [`ModelDescriptor`].
#'
#' @section Input and Output Channels:
#'
#' @examples
#' g = top("input") %>>%
#'   top("select", items = "num") %>>%
#'   top("output") %>>%
#'   top("loss", "mse")
#'
#' task = tsk("iris")
#' g$train(task)
#' @param loss ([`TorchLoss`])\cr
#'   The loss function.
#' @template param_id
#' @template param_param_vals
#' @seealso PipeOpTorch, PipeOpTorchOptimizer
#' @export
PipeOpTorchLoss = R6Class("PipeOpTorchLoss",
  inherit = PipeOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
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
