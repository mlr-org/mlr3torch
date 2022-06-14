#' @title Dropout Layer
#' @description
#' Dropout layer.
#'
#' @export
TorchOpDropout = R6Class("TorchOpDropout",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
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
    .build = function(inputs, param_vals, task) {
      p = param_vals[["p"]] %??% 0.5
      inplace = param_vals[["inplace"]] %??% FALSE
      layer = nn_dropout(p, inplace)
      return(layer)
    }
  )
)
