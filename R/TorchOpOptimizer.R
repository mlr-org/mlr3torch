#' @export
TorchOpOptimizer = R6Class("TorchOpOptimizer",
  inherit = TorchOp,
  public = list(
    initialize = function(id = .optimizer, param_vals = list(), .optimizer) {
      assert_choice(.optimizer, torch_reflections$optimizer)
      param_set = paramsets_optim$get(.optimizer)
      private$.optimizer = .optimizer
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )

    }
  ),
  private = list(
    .train = function(inputs) {
      inputs$input[["optimizer"]] = private$.optimizer
      inputs$input[["optim_args"]] = self$param_set$get_values(tags = "train")
      return(inputs)
    },
    .optimizer = NULL
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("optimizer", TorchOpOptimizer)
