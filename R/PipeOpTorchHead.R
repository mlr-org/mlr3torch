#' @title Output Head
#'
#' @templateVar id nn_head
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @description
#' Output head for classification and regresssion.
#'
#' **NOTE**
#' Because the method `$shapes_out()` does not have access to the task, it returns `c(NA, NA)`.
#' When this [`PipeOp`][mlr3pipelines::PipeOp] is trained however, the model descriptor has the correct output shape.
#'
#' @section Parameters:
#' * `bias` :: `logical(1)`\cr
#'   Whether to use a bias. Default is `TRUE`.
#'
#' @section Internals:
#' Calls [`torch::nn_linear()`] with the input and output features inferred from the input shape / task.
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
      assert_true(length(shapes_in[[1]]) == 2L)
      d = get_nout(task)
      list(c(shapes_in[[1]][[1]], d))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$in_features = shapes_in[[1L]][2L]

      param_vals$out_features = get_nout(task)

      param_vals
    }
  )
)

#' @include zzz.R
register_po("nn_head", PipeOpTorchHead)
