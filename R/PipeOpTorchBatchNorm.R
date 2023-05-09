PipeOpTorchBatchNorm = R6Class("PipeOpTorchBatchNorm",
  inherit = PipeOpTorch,
  public = list(
    # @description
    # Creates a new instance of this [R6][R6::R6Class] class.
    # @template params_pipelines
    # @template param_module_generator
    # @param min_dim (integer(1))\cr
    #   The minimum number of dimension for the input tensor.
    # @param max_dim (`integer(1)`)\cr
    #   The maximum number of dimension for the input tensor.
    initialize = function(id, module_generator, min_dim, max_dim, param_vals = list()) {
      private$.min_dim = assert_int(min_dim, lower = 1)
      private$.max_dim = assert_int(max_dim, lower = 1)
      param_set = ps(
        eps = p_dbl(default = 1e-05, lower = 0, tags = "train"),
        momentum = p_dbl(default = 0.1, lower = 0, tags = "train"),
        affine = p_lgl(default = TRUE, tags = "train"),
        track_running_stats = p_lgl(default = TRUE, tags = "train")
      )

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = module_generator
      )
    }
  ),
  private = list(
    .min_dim = NULL,
    .max_dim = NULL,
    .shapes_out = function(shapes_in, param_vals, task) {
      list(assert_numeric(shapes_in[[1]], min.len = private$.min_dim, max.len = private$.max_dim))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$num_features = shapes_in[[1L]][2L]
      param_vals
    }
  )
)

#' @title 1D Batch Normalization
#'
#' @templateVar id nn_batch_norm1d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_batch_norm description
#'
#' @section Parameters:
#' * `eps` :: `numeric(1)`\cr
#'   A value added to the denominator for numerical stability. Default: `1e-5`.
#' * `momentum` :: `numeric(1)`\cr
#'   The value used for the running_mean and running_var computation. Can be set to `NULL` for cumulative moving average
#'   (i.e. simple average). Default: 0.1
#' * `affine` :: `logical(1)`\cr
#'   a boolean value that when set to `TRUE`, this module has learnable affine parameters. Default: `TRUE`
#' * `track_running_stats` :: `logical(1)`\cr
#'   a boolean value that when set to `TRUE`, this module tracks the running mean and variance, and when set to `FALSE`,
#'   this module does not track such statistics and always uses batch statistics in both training and eval modes.
#'   Default: `TRUE`
#'
#' @section Internals:
#' Calls [`torch::nn_batch_norm1d()`].
#' The parameter `num_features` is inferred as the second dimension of the input shape.
#' @export
PipeOpTorchBatchNorm1D = R6Class("PipeOpTorchBatchNorm1D", inherit = PipeOpTorchBatchNorm,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_batch_norm1d", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_batch_norm1d, min_dim = 2, max_dim = 3, param_vals = param_vals)
    }
  )
)

#' @title 2D Batch Normalization
#'
#' @templateVar id nn_batch_norm2d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_batch_norm description
#'
#' @inheritSection mlr_pipeops_nn_batch_norm1d Parameters
#'
#' @section Internals:
#' Calls [`torch::nn_batch_norm2d()`].
#' The parameter `num_features` is inferred as the second dimension of the input shape.
#' @export
PipeOpTorchBatchNorm2D = R6Class("PipeOpTorchBatchNorm2D", inherit = PipeOpTorchBatchNorm,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_batch_norm2d", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_batch_norm2d, min_dim = 4, max_dim = 4, param_vals = param_vals)
    }
  )
)

#' @title 3D Batch Normalization
#'
#' @templateVar id nn_batch_norm3d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_batch_norm description
#'
#' @inheritSection mlr_pipeops_nn_batch_norm1d Parameters
#'
#' @section Internals:
#' Calls [`torch::nn_batch_norm3d()`].
#' The parameter `num_features` is inferred as the second dimension of the input shape.
#' @export
PipeOpTorchBatchNorm3D = R6Class("PipeOpTorchBatchNorm3D", inherit = PipeOpTorchBatchNorm,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_batch_norm3d", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_batch_norm3d, min_dim = 5, max_dim = 5, param_vals = param_vals)
    }
  )
)

#' @include zzz.R
register_po("nn_batch_norm1d", PipeOpTorchBatchNorm1D)
register_po("nn_batch_norm2d", PipeOpTorchBatchNorm2D)
register_po("nn_batch_norm3d", PipeOpTorchBatchNorm3D)
