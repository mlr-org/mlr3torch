#' @section Order of Inputs
#' Addition and multiplication are albeit not computationally commutative and associative
#' comutative, meaning the order in which they are executed is (mathematically) irrelevant.
#' This however is not the case when stacking tensors. If for some reason the order of concatenation
#' is relevant for your model, it has to be created manually.
#' The order in which the tensors are concatenated is [input1, input2, input3]
#' @example
#' graph = top("fork", .outnum = 2L) %>>%
#'   gunion(list(a = top("linear", out_features = 10L), b = top("linear", out_features = 10L)))
#' graph$add_pipeop(top("merge", .innum = 2L, method = "stack"))
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
    initialize = function(id = "merge", param_vals = list(), .innum) {
      param_set = ps(
        method = p_fct(levels = c("add", "mul", "stack"))
      )
      input = data.table(
        name = paste0("input", seq_len(.innum)),
        train = rep("*", times = .innum),
        predict = rep("*", times = .innum)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      architecture = inputs[[1L]][["architecture"]]$clone(deep = FALSE)
      architecture$ptr = self$id
      ptrs = map(map(inputs[[1L]], "architecture"), "ptr")
      architecture$add_node(self$id, private$.build)
      # TODO: input checks: either task or list(task, architecture)
      if (!is.null(self$state)) { # this means the architecture is already built
        return(inputs)
      }
      if (test_r6(inputs[[1L]], "Task")) { # this means this torchop is the first in the architecture
        # and we have to build the architecture
        task = inputs[[1L]]
        architecture = Architecture$new()
      } else {
        task = inputs[["input"]][["task"]]
        architecture = inputs[["input"]][["architecture"]]
      }

    },
    #' @description builds the merger
    #' @param input a list of tensors
    #' @param param_vals parameter values
    #' @param task the task
    #' @param y the target
    .build = function(input, param_vals, task, y) {
      # input are various tensors
      method = param_vals$method
      shapes = map(input, dim)
      if (method %in% c("add", "multiply")) {
        assert_true(length(unique(shapes)) == 1L)
      }
      if (method == "concat") {
        shapes_wo_last = map(shapes, function(x) x[-length(x)])
        assert_true(length(unique(shapes_wo_last)) == 1L)
      }

      layer = switch(method,
        add = nn_reduce_sum$new(),
        multiply = nn_reduce_multiply$new(),
        concat = nn_reduce_concat$new(dim = length(shapes[[1L]]))
      )
      return(layer)
    }
  )
)


nn_reduce_multiply = nn_module(
  initialize = function() NULL,
  forward = function(inputs) {
    torch_prod(torch_stack(input), dim = 1L)
  }
)

nn_reduce_sum = nn_module(
  initialize = function() NULL,
  forward = function(inputs) {
    torch_sum(torch_stack(input), dim = 1L)
  }
)

nn_reduce_concat = nn_module(
  initialize = function(dim) self$dim = dim,
  forward = function(inputs) {
    torch_stack(inputs, dim = self$dim)
  }
)

#' @include mlr_torchops.R
mlr_torchops$add("merge", TorchOpMerge)
