#' @title TorchOpSoftmax
#' @include TorchOpSoftmax.R
#' @export
TorchOpSoftmax = R6::R6Class("TorchOpSoftmax",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "softmax", param_vals = list()) {
      param_set = ps(
        dim = p_int(1L, Inf, tags = c("train", "required"))
      )
      super$initialize(
        id = id,
        module_generator = nn_softmax,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) shapes_in,
    .shape_dependent_params = function(shapes_in) list()
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("softmax", value = TorchOpSoftmax)
