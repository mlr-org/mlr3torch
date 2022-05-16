#' @title Linear TorchOp
#' @include TorchOpLinear.R
#' @section Dimensions:
#' (n, ..., in_features) --> (n, ..., out_features)
#' @export
TorchOpLinear = R6Class("TorchOpLinear",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @parm param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "linear", param_vals = list()) {
      param_set = ps(
        out_features = p_int(1L, Inf, tags = c("train", "required")),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "linear",
    .build = function(inputs, param_vals, task, y) {
      input = inputs$input
      # TODO: Define a clean interface what dimensions a TorchOp requires as input and what
      # it then outputs
      in_features = input$shape[length(input$shape)]
      layer = invoke(nn_linear, in_features = in_features, .args = param_vals)
      return(layer)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("linear", value = TorchOpLinear)
