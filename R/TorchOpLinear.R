#' @title Linear TorchOp
#' @description
#' Standard linear layer.
#'
#' @section Calls:
#' Calls `torch::nn_linear()`.
#'
#' @section Custom mlr3 parameters:
#' * `in_channels` - This parameter is inferred as the last dimension of the input tensor.
#'
#' @template param_id
#' @template param_param_vals
#'
#'
#' @export
PipeOpTorchLinear = R6Class("PipeOpTorchLinear",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_linear", param_vals = list()) {
      param_set = ps(
        out_features = p_int(1L, Inf, tags = c("train", "required")),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_linear
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in) list(in_features = shapes_in[[1]]),
    .shapes_out = function(shapes_in, param_vals) list(head(shapes_in[[1]], -1), param_vals$out_features)
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("nn_linear", value = PipeOpTorchLinear)
