#' @title Merges Multiple Tensors
#'
#' @description
#' Merges multiple tensors.
#'
#' @template param_id
#' @template param_param_vals
#' @param .method (`character(1)`)\cr
#'   The method for the concatenation. One of "add", "mul" or "cat".
#' @template param_.innum
#'
#' @section Order of inputs:
#' In case the order of input matters (e.g. for method "cat"), the constructor argument 'innum'
#' should be set.
#' @export
#'
#' @examples
#'
#' top1 = top("linear_1", out_features = 10L)
#' top2 = top("linear_2", out_features = 10L)
#'
#' # order is not specified:
#' top_add = top("add")
#' graph = gunion(list(top1, top2)) %>>% top_add
#'
#' # order is specified:
#' top_cat = top("cat", .innum = 2)
#' graph = gunion(list(top1, top2))
#' graph$add_pipeop(top_cat)
#' graph$add_edge(src_id = "linear_1", dst_id = "cat", dst_channel = "input2")
#' graph$add_edge(src_id = "linear_2", dst_id = "cat", dst_channel = "input1")
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
          train = "ModelArgs",
          predict = "Task"
        )
      } else {
        input = data.table(
          name = paste0("input", seq_len(.innum)),
          train = rep("ModelArgs", times = .innum),
          predict = rep("Task", times = .innum)
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
        add = nn_merge_sum(dim),
        mul = nn_merge_mul(dim),
        cat = nn_merge_cat(dim)
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
#' @template param_id
#' @template param_param_vals
#' @template param_.innum
#'
#' @export
TorchOpAdd = R6Class("TorchOpAdd",
  inherit = TorchOpMerge,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
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
#' @template param_id
#' @template param_param_vals
#' @template param_.innum
#'
#' @export
TorchOpMul = R6Class("TorchOpMul",
  inherit = TorchOpMerge,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
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
#' @template param_param_vals
#' @template param_.innum
#'
#' @export
TorchOpCat = R6Class("TorchOpCat",
  inherit = TorchOpMerge,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
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
