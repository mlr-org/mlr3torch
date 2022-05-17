#' @title Merges Multiple Tensors
#'
#' @description
#' Merges multiple tensors.
#'
#' @param id (`character(1)`)\cr
#'   The for of the object.
#' @param param_vals (named `list()`)\cr
#'   The initial parameters for the object.
#' @param .method (`character(1)`)\cr
#'   The method for the concatenation. One of "add"
#' @param .innum (`character()`)\cr
#'   Number of inputs (optional). If provided, input channels are set to `"input1"`, `"input2"`,
#'   etc..
#'
#' @section Order of Inputs:
#' Addition and multiplication are albeit not computationally commutative and associative
#' comutative, meaning the order in which they are executed is (mathematically) irrelevant.
#' This however is not the case when stacking tensors. If for some reason the order of stacking.
#' is relevant for your model, it has to be created manually.
#' The order in which the tensors are concatenated is [input1, input2, input3].
#'
#' @export
TorchOpMerge = R6Class("TorchOpMerge",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "merge", param_vals = list(), .method, .innum = NULL) {
      private$.method = assert_choice(.method, c("add", "mul", "cat"))
      param_set = ps(
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
    },
    .method = NULL
  )
)

#' @title Add Torch Tensors
#'
#' @description
#' Add torch tensors.
#'
#' @export
TorchOpAdd = R6Class("TorchOpAdd",
  inherit = TorchOpMerge,
  public = list(
    initialize = function(id = "add", param_vals = list(), .innum = NULL) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .method = "add",
        .innum = .innum
      )
    }
  )
)

#' @title Multiply Torch Tensors
#'
#' @description
#' Concatenate torch tensors.
#'
#' @export
TorchOpMul = R6Class("TorchOpMul",
  inherit = TorchOpMerge,
  public = list(
    initialize = function(id = "mul", param_vals = list(), .innum = NULL) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .method = "mul",
        .innum = .innum
      )
    }
  )
)

#' @title Concatenate Torch Tensors
#'
#' @description
#' Concatenate torch tensors.
#'
#' @export
TorchOpCat = R6Class("TorchOpCat",
  inherit = TorchOpMerge,
  public = list(
    initialize = function(id = "cat", param_vals = list(), .innum = NULL) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .method = "cat",
        .innum = .innum
      )
    }
  )
)

nn_merge_mul = nn_module(
  "merge_multiply",
  initialize = function(dim) self$dim = dim,
  forward = function(...) {
    torch_prod(torch_stack(list(...)), dim = self$dim)
  }
)

nn_merge_sum = nn_module(
  "merge_sum",
  initialize = function(dim) self$dim = dim,
  forward = function(...) {
    torch_sum(torch_stack(list(...)), dim = self$dim)
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
#' @include mlr_torchops.R
mlr_torchops$add("add", TorchOpAdd)
#' @include mlr_torchops.R
mlr_torchops$add("mul", TorchOpMul)
#' @include mlr_torchops.R
mlr_torchops$add("cat", TorchOpCat)
