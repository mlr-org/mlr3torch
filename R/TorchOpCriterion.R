#' @export
TorchOpCriterion = R6Class(
  inherit = TorchOp,
  public = list(
    initialize = function(id = .criterion, param_vals = list(), .criterion) {
      param_set = make_paramset_criterion()
      private$.criterion = .criterion
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      inputs$input[["criterion"]] = private$.criterion
      inputs$input[["criterion_args"]] = self$param_set$get_values(tags = "train")
      return(inputs)
    },
    .criterion = NULL
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("criterion", TorchOpCriterion)
