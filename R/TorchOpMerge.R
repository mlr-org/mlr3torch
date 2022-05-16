#' @title Merges Multiple Tensors
#' @description Merges multiple tensors.
#' @section Order of Inputs:
#' Addition and multiplication are albeit not computationally commutative and associative
#' comutative, meaning the order in which they are executed is (mathematically) irrelevant.
#' This however is not the case when stacking tensors. If for some reason the order of stacking.
#' is relevant for your model, it has to be created manually.
#' The order in which the tensors are concatenated is [input1, input2, input3]
#' @export
TorchOpMerge = R6Class("TorchOpMerge",
  inherit = TorchOp,
  public = list(
    #' @description Initilizes a new instance of this class.\cr
    #' @param id (`character(1)`) The id of the TorchOp.
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The for of the object.
    #' @parm param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    #' @param param_vals (`list()`) List of parameter values.
    #' @param .innum (`character()`) number of inputs.
    #' @description Initializes an object of class [TorchOpInput]
    #' @param id (`character(1)`) The id of the object.
    #' @param param_vals (`named list()`) The parameter values.
    initialize = function(id = "merge", param_vals = list(), .innum = NULL) {
      param_set = ps(
        method = p_fct(levels = c("add", "mul", "cat"), tags = c("train", "required")),
        dim = p_int(tags = "train")
      )
      if (is.null(.innum)) {
        input = data.table(
          name = "...",
          train = "*",
          predict = "*"
        )
      } else {
        input = data.table(
          name = paste0("input", seq_len(.innum)),
          train = rep("*", times = .innum),
          predict = rep("*", times = .innum)
        )
      }
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input
      )
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task, y) {
      # input are various tensors
      method = param_vals$method
      # order of the next two calls is important, because if we overwrite dim first,
      # map(inputs, dim) does not apply the function dim()
      shapes = map(inputs, dim)
      dim = param_vals$dim %??% length(shapes[[1L]])
      # NOTE: no input checking, this is automatically done when the forward function
      # is called afterwards
      layer = switch(method,
        add = nn_merge_sum(),
        mul = nn_merge_mul(),
        cat = nn_merge_cat(dim = dim)
      )
      return(layer)
    }
  )
)



nn_merge_mul = nn_module(
  "merge_multiply",
  initialize = function() NULL,
  forward = function(...) {
    torch_prod(torch_stack(list(...)), dim = 1L)
  }
)

nn_merge_sum = nn_module(
  "merge_sum",
  initialize = function() NULL,
  forward = function(...) {
    torch_sum(torch_stack(list(...)), dim = 1L)
  }
)

nn_merge_cat = nn_module(
  "merge_cat",
  initialize = function(dim) self$dim = dim,
  forward = function(...) {
    torch_cat(list(...), dim = self$dim)
  }
)

#' @include mlr_torchops.R
mlr_torchops$add("merge", TorchOpMerge)
