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
#' @template param_id
#' @template param_param_vals
#'
#' @export
PipeOpTorchBatchNorm = R6Class("PipeOpTorchBatchNorm",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, module_generator, min_dim, max_dim, param_vals = list()) {
      private$.min_dim = assert_int(min_dim, lower = 1)
      private$.max_dim = assert_int(max_dim, min_dim = 1)
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
    .min_dim = NULL,
    .max_dim = NULL,
    .shapes_out = function(shapes_in, param_vals) {
      list(assert_numeric(shapes_in[[1]], min.len = private$.min_dim, max.len = private$.max_dim))
    }
  )
)


PipeOpTorchBatchNorm1D = R6Class("PipeOpTorchBatchNorm1D", inherit = PipeOpTorchBatchNorm,
  public = list(
    initialize = function(id = "nn_batch_norm1d", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_batch_norm1d, min_dim = 2, max_dim = 3, param_vals = param_vals)
    }
  )
)

PipeOpTorchBatchNorm2D = R6Class("PipeOpTorchBatchNorm2D", inherit = PipeOpTorchBatchNorm,
  public = list(
    initialize = function(id = "nn_batch_norm2d", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_batch_norm2d, min_dim = 4, max_dim = 4, param_vals = param_vals)
    }
  )
)

PipeOpTorchBatchNorm3D = R6Class("PipeOpTorchBatchNorm3D", inherit = PipeOpTorchBatchNorm,
  public = list(
    initialize = function(id = "nn_batch_norm3d", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_batch_norm3d, min_dim = 5, max_dim = 5, param_vals = param_vals)
    }
  )
)


#' @include zzz.R
register_po("nn_batch_norm1d", PipeOpTorchBatchNorm1D)
register_po("nn_batch_norm2d", PipeOpTorchBatchNorm1D)
register_po("nn_batch_norm3d", PipeOpTorchBatchNorm1D)


