#' @title Head for Neural Networks.
#' @description
#' Calls `torch::nn_linear()` with the correct parameters.
#'
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchHead = R6Class("PipeOpTorchHead",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = "nn_head", param_vals = list()) {
      param_set = ps(
        bias = p_lgl(default = TRUE, tags = "train")
      )
      param_set$values = list(bias = TRUE)

      super$initialize(
        module_generator = NULL,
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        inname = "input"
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      assert_true(length(shapes_in[[1]]) == 2L)
      list(c(shapes_in[[1]][[1]], NA_integer_))
    },
    .shape_dependent_params = function(shapes_in, param_vals) {
      c(param_vals, list(in_features = shapes_in[[1]][[2]]))
    },
    .train = function(inputs) {
      param_vals = self$param_set$get_values()

      task = inputs[[1]]$task
      param_vals$out_features = switch(task$task_type,
        classif = length(task$class_names),
        regr = 1,
        stopf("Task type not supported!")
      )

      PipeOpTorchLinear$new(id = self$id, param_vals = param_vals)$train(inputs)
    }
  )
)

#' @include zzz.R
register_po("nn_head", PipeOpTorchHead)
