#' @title Base Class for Batch Normalization
#'
#' @usage NULL
#' @name mlr_pipeops_torch_batch_norm
#' @format `r roxy_pipeop_torch_format()`
#'
#' @description
#' Base class for batch normalization.
#' Don't use this class directly.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchBatchNorm)`
#' * `r roxy_param_id()`
#' * `r roxy_param_param_vals()`
#' * `r roxy_param_module_generator()`
#' * `min_dim` :: `integer(1)`\cr
#'   The minimum number of dimension for the input tensor.
#' * `max_dim` :: `integer(1)`\cr
#'   The maximum number of dimension for the input tensor.
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
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
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' See the respective child class.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchBatchNorm = R6Class("PipeOpTorchBatchNorm",
  inherit = PipeOpTorch,
  public = list(
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

#' @title 1D Batch Normalization
#'
#' @usage NULL
#' @name mlr_pipeops_torch_batch_norm
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_batch_norm1d description
#'
#' @description
#' Base class for batch normalization.
#' Don't use this class directly.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchBatchNorm1D)`
#' * `r roxy_param_id("nn_batch_norm_1d")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
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
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' See the respective child class.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchBatchNorm1D = R6Class("PipeOpTorchBatchNorm1D", inherit = PipeOpTorchBatchNorm,
  public = list(
    initialize = function(id = "nn_batch_norm1d", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_batch_norm1d, min_dim = 2, max_dim = 3, param_vals = param_vals)
    }
  )
)

#' @title 2D Batch Normalization
#'
#' @usage NULL
#' @name mlr_pipeops_torch_batch_norm
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_batch_norm2d description
#'
#' @section Construction: `r roxy_construction(PipeOpTorchBatchNorm2D)`
#' * `r roxy_param_id("nn_batch_norm_2d")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
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
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' See the respective child class.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchBatchNorm2D = R6Class("PipeOpTorchBatchNorm2D", inherit = PipeOpTorchBatchNorm,
  public = list(
    initialize = function(id = "nn_batch_norm2d", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_batch_norm2d, min_dim = 4, max_dim = 4, param_vals = param_vals)
    }
  )
)

#' @title 3D Batch Normalization
#'
#' @usage NULL
#' @name mlr_pipeops_torch_batch_norm
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_batch_norm3d description
#'
#' @section Construction: `r roxy_construction(PipeOpTorchBatchNorm3D)`
#' * `r roxy_param_id("nn_batch_norm_3d")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
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
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' See the respective child class.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
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


