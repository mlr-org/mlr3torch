#' @export
TorchOpLoss = R6Class("TorchOpLoss",
  inherit = TorchOp,
  public = list(
    initialize = function(id = loss, param_vals = list(), loss) {
      assert_choice(loss, unlist(torch_reflections$loss))
      param_set = paramsets_loss$get(loss)
      private$.loss = loss
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )

    }
  ),
  private = list(
    .train = function(inputs) {
      inputs$input[["loss"]] = private$.loss
      inputs$input[["loss_args"]] = self$param_set$get_values(tags = "train")
      return(inputs)
    },
    .loss = NULL
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("loss", TorchOpLoss)
