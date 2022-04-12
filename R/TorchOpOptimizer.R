#' @export
TorchOpOptimizer = R6Class(
  inherit = TorchOp,
  public = list(
    initialize = function(id = .optimizer, param_vals = list(), .optimizer) {
      param_set = make_paramset_optim(.optimizer)
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
      inputs$input[["optimizer_args"]] = self$param_set$get_values(tags = "train")
      return(inputs)
    },
    .optimizer = NULL
  )
)

mlr_torchops$add("optimizer", TorchOpOptimizer)

if (FALSE) {
  opt = top("optimizer", optimizer = "adam", lr = 0.1)

}
