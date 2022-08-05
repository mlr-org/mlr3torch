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
TorchOpLinear = R6Class("TorchOpLinear",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "linear", param_vals = list()) {
      param_set = ps(
        out_features = p_int(1L, Inf, tags = c("train", "required")),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      input = inputs$input
      assert_true(length(input$shape) >= 2L)
      # TODO: Define a clean interface what dimensions a TorchOp requires as input and what
      # it then outputs
      in_features = input$shape[length(input$shape)]
      args = insert_named(param_vals, list(in_features = in_features))

      invoke(nn_linear, .args = args)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("linear", value = TorchOpLinear)
