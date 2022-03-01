TorchOpDropout = R6Class("TorchOpDropout",
  inherit = TorchOp,
  public = list(
    intialize = function(id = "dropout", param_vals = list()) {
      param_set = ps(
        p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "dropout",
    .build = function(input, param_vals, task) {
      p = param_vals[["p"]] %??% 0.5
      inplace = param_vals[["inplace"]] %??% FALSE
      layer = nn_dropout(p, inplace)
      return(layer)
    }
  )
)
