#' @title Output Head
#' @description
#' Output head for classification and regresssion.
#'
#' @section nn_module:
#' Calls [`torch::nn_linear()`] with the input and output features inferred from the input shape / task.
#' For
#' * binary classification, the output dimension is 1.
#' * multiclass classification, the output dimension is the number of classes.
#' * regression, the output dimension is 1.
#'
#' @section Parameters:
#' * `bias` :: `logical(1)`\cr
#'   Whether to use a bias. Default is `TRUE`.
#' @details
#' When the method `$shapes_out()` does not have access to the task, it returns `c(NA, NA)`.
#' When this [`PipeOp`][mlr3pipelines::PipeOp] is trained however, the model descriptor has the correct output shape.
#'
#' @templateVar id nn_head
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchHead = R6Class("PipeOpTorchHead",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_head", param_vals = list()) {
      param_set = ps(bias = p_lgl(default = TRUE, tags = "train"))
      super$initialize(
        module_generator = nn_linear,
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        inname = "input"
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      if (length(shapes_in[[1]]) != 2L) {
        stopf("PipeOpTorchHead expects 2D input, but got %s.", shape_to_str(shapes_in))
      }
      d = output_dim_for(task)
      list(c(shapes_in[[1]][[1]], d))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$in_features = shapes_in[[1L]][2L]

      param_vals$out_features = output_dim_for(task)

      param_vals
    }
  )
)

#' @include aaa.R
register_po("nn_head", PipeOpTorchHead)
