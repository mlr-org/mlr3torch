PipeOpReLU = R6Class("PipeOpReLU",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    initialize = function(id = "relu", param_vals = list()) {
      param_set = ps()
    }
  )
)
