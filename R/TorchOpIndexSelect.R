#' if we are only interested in the CLS Token.
#' @export
TorchOpIndexSelect = R6Class("TorchOpIndexSelect",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @parm param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "indexselect", param_vals = list()) {
      param_set = ps(
        dim = p_int(default = 2L, lower = 0L),
        index = p_int(lower = 0)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "indexselect",
    .build = function(inputs, param_vals, task, y) {
      input = inputs$input
      dim = param_vals[["dim"]] %??% 2L
      index = param_vals[["index"]] %??% input$shape[[dim]]
      layer = nn_module(
        "indexselect",
        initialize = function(dim, index) {
          self$dim = dim
          self$index = index
        },
        forward = function(input) {
          torch_index_select(input, self$dim, torch_tensor(self$index, dtype = torch_int()))
        }
      )(dim = dim, index = index)
      return(layer)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("indexselect", TorchOpIndexSelect)
