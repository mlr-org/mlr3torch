#' @title Head
#'
#' @usage NULL
#' @name mlr_pipeops_torch_head
#' @format `r roxy_pipeop_torch_format()`
#'
#' @description
#' Output head for classification and regresssion.
#'
#' @section Construction:
#' ```
#' PipeOpTorchHead$new(id = "nn_head", param_vals = list())
#' ```
#' `r roxy_param_id("Modulenn_head")`
#' `r roxy_param_param_vals()`
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
#' Calls [`torch::nn_linear()`] with the output dimension inferred from the task.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_head")
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 10))
#'
#' # pot
#' obj = pot("head")
#' obj$id
#'
PipeOpTorchHead = R6Class("PipeOpTorchHead",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_head", param_vals = list()) {
      param_set = ps(bias = p_lgl(default = TRUE, tags = "train"))
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
