#' @title Softmax
#'
#' @usage NULL
#' @template pipeop_torch_format
#'
#' @inherit torch::nn_softmax description
#'
#' @description
#'
#' @section Torch Module:
#' Wraps [`torch::nn_softmax`].
#'
#'
#' @template torch_license_docu
#' @template param_id
#' @template param_param_vals
#'
#' @export
PipeOpTorchSoftmax = R6::R6Class("PipeOpTorchSoftmax",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "nn_softmax", param_vals = list()) {
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
  )
)

#' @include zzz.R
register_po("nn_softmax", PipeOpTorchSoftmax)
