#' @title Output Head
#'
#' @usage NULL
#' @name mlr_pipeops_torch_head
#' @format `r roxy_format(PipeOpTorchHead)`
#'
#' @description
#' Output head for classification and regresssion.
#'
#' **NOTE**
#' Because the method `$shapes_out()` does not have access to the task, it returns `c(NA, NA)`.
#' When this [`PipeOp`] is trained however, the model descriptor has the correct output shape.
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchHead)`
#' * `r roxy_param_id("nn_head")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `bias` :: `logical(1)`\cr
#'   Whether to use a bias. Default is `TRUE`.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`torch::nn_linear()`] with the input and output features inferred from the input shape / task.
#' @section Credit: `r roxy_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#'obj = po("nn_head")
#'obj$id
#'obj$module_generator
#'obj$shapes_out(c(16, 10), tsk("iris"))
#'obj$shapes_out(c(16, 10), tsk("mtcars"))
PipeOpTorchHead = R6Class("PipeOpTorchHead",
  inherit = PipeOpTorch,
  public = list(
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
      d = switch(task$task_type,
        regr = 1,
        classif = length(task$class_names),
        stopf("Task type not supported")
      )
      list(c(shapes_in[[1]][[1]], d))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$in_features = shapes_in[[1L]][2L]
      param_vals$out_features = switch(task$task_type,
        classif = length(task$class_names),
        regr = 1,
        stopf("Task type not supported!")
      )

      param_vals
    }
  )
)

#' @include zzz.R
register_po("nn_head", PipeOpTorchHead)
