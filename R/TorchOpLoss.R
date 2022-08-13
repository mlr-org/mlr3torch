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
TorchOpLoss = R6Class("TorchOpLoss",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(optimizer, id = "loss", param_vals = list()) {
      assert_r6(optimizer, "TorchLoss")
      private$.loss = optimizer
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
    .shapes_out = function(shapes_in, param_vals) shapes_in,
    .shape_dependent_params = function(shapes_in) list(),
    .loss = NULL
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("loss", TorchOpLoss)
