#' @title Batch Normalization
#' @description
#' Batch normalization.
#' @section Inferred Parameters:
#' Depending on the shape of the input, the corresponding batch-norm is chosen.
#'
#'  * 2d input: 1d
#'  * 3d input: 1d
#'  * 4d input: 2d
#'  * 5d input: 3d
#'
#' * The parameter `num_features` is automatically inferred from the input as the size of the
#' second dimension.
#'
#' @section Calls:
#' `nn_batch_norm1d()`, `nn_batch_norm2d()` or `nn_batch_norm3d()` depending on the input.
#'
#' @section References:
#' * r format_bib("ioffe2015batch")`
#'
#' @export
TorchOpBatchNorm = R6Class("TorchOpBatchNorm1D",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "batch_norm", param_vals = list()) {
      param_set = ps(
        eps = p_dbl(default = 1e-05, lower = 0, tags = "train"),
        momentum = p_dbl(default = 0.1, lower = 0, tags = "train"),
        affine = p_lgl(default = TRUE, tags = "train"),
        track_running_stats = p_lgl(default = TRUE, tags = "train")
      )

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task, y) {
      input = inputs$input
      assert_integer(length(input$shape), lower = 2L, upper = 5L)

      num_features = input$shape[2L]
      class = switch(length(input$shape),
        NULL,
        torch::nn_batch_norm1d,
        torch::nn_batch_norm1d,
        torch::nn_batch_norm2d,
        torch::nn_batch_norm3d
      )

      layer = invoke(
        class$new,
        num_features = num_features,
        .args = param_vals
      )

      return(layer)
    }
  )
)


#' @include mlr_torchops.R
mlr_torchops$add("batch_norm", TorchOpBatchNorm)
