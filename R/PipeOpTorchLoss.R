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
      private$.loss = assert_r6(as_torch_loss(loss), "TorchLoss")
      super$initialize(
        id = id,
        param_set = alist(private$.loss$param_set),
        param_vals = param_vals,
        module_generator = NULL
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
