#' @title Softmax
#' @inherit torch::nnf_softmax description
#' @section nn_module:
#' Calls [`torch::nn_softmax()`] when trained.
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
#'
#' @templateVar id nn_softmax
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchSoftmax = R6::R6Class("PipeOpTorchSoftmax",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_softmax", param_vals = list()) {
      param_set = ps(
        # negative values count down from the last dimension
        dim = p_int(tags = c("train", "required"))
      )
      super$initialize(
        id = id,
        module_generator = nn_softmax,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      dim = param_vals[["dim"]]
      assert_dim_in_range(dim, resolve_dim(dim, shape), shape, self$id)
      shapes_in
    }
  )
)

#' @include aaa.R
register_po("nn_softmax", PipeOpTorchSoftmax)
