#' @section Order of Inputs
#' Addition and multiplication are albeit not computationally commutative and associative
#' comutative, meaning the order in which they are executed is (mathematically) irrelevant.
#' This however is not the case when stacking tensors. If for some reason the order of stacking.
#' is relevant for your model, it has to be created manually.
#' The order in which the tensors are concatenated is [input1, input2, input3]
#' @example
#' graph = top("fork", .outnum = 2L) %>>%
#'   gunion(list(a = top("linear", out_features = 10L), b = top("linear", out_features = 10L)))
#' graph$add_pipeop(top("merge", .innum = 2L, method = "cat"))
#' graph$add_edge("a.linear", "merge", dst_channel = "input2")
#' graph$add_edge("b.linear", "merge", dst_channel = "input1")
#' graph$edges
#'
TorchOpMerge = R6Class("TorchOpMerge",
  inherit = TorchOp,
  public = list(
    #' @description Initilizes a new instance of this class.
    #' @param id (`character(1)`) The id of the TorchOp.
    #' @param param_vals (`list()`) List of parameter values.
    #' @param .innum (`character()`) number of inputs.
    initialize = function(id = "merge", param_vals = list(), .innum = NULL) {
      param_set = ps(
        method = p_fct(levels = c("add", "mul", "cat"), tags = "train"),
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
    #' @description builds the merger
    #' @param input a list of tensors
    #' @param param_vals parameter values
    #' @param task the task
    #' @param y the target
    .build = function(inputs, param_vals, task, y) {
      # input are various tensors
      method = param_vals$method
      # order of the next two calls is important, because if we overwrite dim first,
      # map(inputs, dim) does not apply the function dim()
      shapes = map(inputs, dim)
      dim = param_vals$dim %??% length(shapes[[1L]])
      assert_true(dim <= length(shapes[[1L]]))

      if (method %in% c("add", "mul")) {
        assert_true(length(unique(shapes)) == 1L)
      }
      if (method == "cat") {
        shapes_wo_dim = map(shapes, function(x) x[-dim])
        assert_true(length(unique(shapes_wo_dim)) == 1L)
      }

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
