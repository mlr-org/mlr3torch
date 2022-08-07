#' @title Flattens Tensor
#' @description
#' Flattens a tensor
#' @section Calls:
#' Calls `nn_flatten()`
#'
#' @export
TorchOpFlatten = R6Class(
  inherit = PipeOpTorch,
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
        param_vals = param_vals,
        module_generator = nn_flatten
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in) list(),
    .shapes_out = function(shapes_in, param_vals) {
      shape = shapes_in[[1]]
      start_dim = param_vals$start_dim %??% 2
      end_dim = param_vals$end_dim %??% 2

      if (start_dim < 0) start_dim = 1 + length(shape) + start_dim
      if (end_dim < 0) end_dim = 1 + length(shape) + end_dim
      assert_int(start_dim, lower = 1, upper = length(shape))
      assert_int(end_dim, lower = start_dim, upper = length(shape))

      list(c(shape[seq_len(start_dim - 1)], prod(shape[start_dim:end_dim]), shape[seq_len(length(shape) - end_dim) + end_dim]))
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("flatten", TorchOpFlatten)
