#' @title Identity Layer
#' @inherit torch::nn_identity description
#' @section nn_module:
#' Calls [`torch::nn_identity()`] when trained, which passes the input unchanged to the output.
#'
#' @templateVar id nn_identity
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchIdentity = R6Class("PipeOpTorchIdentity",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_identity", param_vals = list()) {
      param_set = ps()
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_identity
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      shapes_in
    }
  )
)

#' @include aaa.R
register_po("nn_identity", PipeOpTorchIdentity)