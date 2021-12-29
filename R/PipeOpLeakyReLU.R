#' @export
PipeOpReLU = R6Class("PipeOpReLU",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    initialize = function(id = "leakyrelu", param_vals = list()) {
      param_set = ps(
        negative_slope = p_dbl(lower = 0, upper = Inf, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        packages = c("mlr3torch", "torch")
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      task = inputs[["task"]]
    }
  )

)

if (FALSE) {
  po = PipeOpReLU$new()
}
