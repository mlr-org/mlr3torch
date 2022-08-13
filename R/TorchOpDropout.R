#' @title Dropout Layer
#' @description
#' Dropout layer.
#'
#' @export
PipeOpTorchDropout = R6Class("PipeOpTorchDropout",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "nn_dropout", param_vals = list()) {
      param_set = ps(
        p = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_dropout
      )
    }
  )
)

#' @include mlr_torchops.R
register_po("nn_dropout", PipeOpTorchDropout)
