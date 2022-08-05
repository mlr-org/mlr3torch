#' @title Reshape Operations
#' @description
#' Reshapes a tensor to the given shape or squeezes / unsqueezes a tensor for the given dim
#' @name reshape_ops
NULL

#' @template param_id
#' @template param_param_vals
#' @rdname reshape_ops
#' @export
TorchOpReshape = R6Class("TorchOpReshape",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "reshape", param_vals = list()) {
      param_set = ps(
        shape = p_uty(tags = c("train", "required"), custom_check = check_integerish)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inpus, task) {
      param_vals = self$param_set$get_values(tags = "train")
      invoke(nn_reshape, .args = param_vals)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname reshape_ops
#' @export
TorchOpSqueeze = R6Class("TorchOpSqueeze",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "squeeze", param_vals = list()) {
      param_set = ps(
        dim = p_uty(tags = c("train", "required"), custom_check = check_integerish)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inpus, task) {
      param_vals = self$param_set$get_values(tags = "train")
      invoke(nn_squeeze, .args = param_vals)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname reshape_ops
#' @export
TorchOpUnsqueeze = R6Class("TorchOpUnqueeze",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "unsqueeze", param_vals = list()) {
      param_set = ps(
        dim = p_uty(tags = c("train", "required"), custom_check = check_integerish)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inpus, task) {
      param_vals = self$param_set$get_values(tags = "train")
      invoke(nn_unsqueeze, .args = param_vals)
    }
  )
)

nn_reshape = nn_module(
  initialize = function(shape) {
    self$shape = shape
  },
  forward = function(input) {
    input$reshape(self$shape)
  }
)

nn_squeeze = nn_module(
  initialize = function(dim) {
    self$dim = dim
  },
  forward = function(input) {
    input$squeeze(self$dim)
  }
)

nn_unsqueeze = nn_module(
  initialize = function(dim) {
    self$dim = dim
  },
  forward = function(input) {
    input$unsqueeze(self$dim)
  }
)

#' @include mlr_torchops.R
mlr_torchops$add("reshape", TorchOpReshape)

#' @include mlr_torchops.R
mlr_torchops$add("unsqueeze", TorchOpUnsqueeze)

#' @include mlr_torchops.R
mlr_torchops$add("squeeze", TorchOpSqueeze)
