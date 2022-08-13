#' @title Loss Function
#' @examples
#' g = top("input") %>>%
#'   top("select", items = "num") %>>%
#'   top("output") %>>%
#'   top("loss", "mse")
#'
#' task = tsk("iris")
#' g$train(task)
#' @export
PipeOpTorchLoss = R6Class("PipeOpTorchLoss",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(loss, id = "torch_loss", param_vals = list()) {
      assert_r6(loss, "TorchLoss")
      private$.loss = loss
      super$initialize(
        id = id,
        param_set = alist(private$.optimizer$param_set),
        param_vals = param_vals
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
