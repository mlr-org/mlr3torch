#' @title Output Layer for a Neural Network
#' @description
#' Calls `torch::nn_linear()` with the correct parameters.
#' @section Input:
#' This TorchOp expects as input a 2 dimensional torch tensor when building.
#' Creates a output layer for the given task
#' @export
TorchOpOutput = R6Class("TorchOpOutput",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "output", param_vals = list()) {
      param_set = ps(
        bias = p_lgl(default = TRUE, tags = "train")
      )
      param_set$values = list(bias = TRUE)

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      assert_true(length(shapes_in[[1]]) == 2L)
      list(shapes_in[[1]][[1]], NA_integer_)
    },
    .shape_dependent_params = function(shapes_in) {
      list(in_features = shapes_in[[1]][[2]])
    },
    .train = function(inputs) {
      param_vals = self$param_set$get_values()

      task = inputs[[1]]$task
      out_features = switch(task$task_type,
        classif = length(task$class_names),
        regr = 1,
        stopf("Task type not supported!")
      )

      TorchOpLinear$new(out_features = out_features, param_vals = param_vals)$train(inputs)
    }
  )
)

mlr_torchops$add("output", TorchOpOutput)
