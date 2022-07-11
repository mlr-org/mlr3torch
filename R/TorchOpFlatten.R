#' @title Flattens Tensor
#' @description
#' Flattens a tensor
#' @section Calls:
#' Calls `nn_flatten()`
#'
#' @export
TorchOpFlatten = R6Class(
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "flatten", param_vals = list()) {
      param_set = ps(
        start_dim = p_int(default = 2L, lower = 1L, tags = "train"),
        end_dim = p_int(default = -1L, lower = 1L, tags = "train", special_vals = list(-1L))
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      invoke(nn_flatten, .args = param_vals)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("flatten", TorchOpFlatten)
