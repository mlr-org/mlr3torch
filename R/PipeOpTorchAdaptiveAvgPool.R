PipeOpTorchAdaptiveAvgPool = R6Class("PipeOpTorchAdaptiveAvgPool",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id, d, param_vals = list()) {
      private$.d = assert_int(d, lower = 1, upper = 3)
      module_generator = switch(d, nn_adaptive_avg_pool1d, nn_adaptive_avg_pool2d, nn_adaptive_avg_pool3d)
      check_vector = make_check_vector(private$.d)
      param_set = ps(
        output_size = p_uty(custom_check = check_vector, tags = c("required", "train"))
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
    .additional_phash_input = function() {
      list(private$.d)
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      list(adaptive_avg_output_shape(
        shape_in = shapes_in[[1]],
        conv_dim = private$.d,
        output_size = param_vals$output_size
      ))
    },
    .d = NULL
  )
)

adaptive_avg_output_shape = function(shape_in, conv_dim, output_size) {
  shape_in = assert_integerish(shape_in, min.len = conv_dim, coerce = TRUE)

  if (length(output_size) == 1) output_size = rep(output_size, conv_dim)

  shape_head = utils::head(shape_in, -conv_dim)
  if (length(shape_head) <= 1) warningf("Input tensor does not have batch dimension.")

  shape_tail = output_size

  c(shape_head, shape_tail)
}

#' @title 1D Adaptive Average Pooling
#'
#' @inherit torch::nnf_adaptive_avg_pool1d description
#' @section nn_module:
#' Calls [`nn_adaptive_avg_pool1d()`][torch::nn_adaptive_avg_pool1d] during training.
#' @section Parameters:
#' * `output_size` :: `integer(1)`\cr
#'   The target output size. A single number.
#' @templateVar id nn_adaptive_avg_pool1d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchAdaptiveAvgPool1D = R6Class("PipeOpTorchAdaptiveAvgPool1D", inherit = PipeOpTorchAdaptiveAvgPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_adaptive_avg_pool1d", param_vals = list()) {
      super$initialize(id = id, d = 1, param_vals = param_vals)
    }
  )
)

#' @title 2D Adaptive Average Pooling
#'
#' @inherit torch::nnf_adaptive_avg_pool2d description
#'
#' @section nn_module:
#' Calls [`nn_adaptive_avg_pool2d()`][torch::nn_adaptive_avg_pool2d] during training.
#' @section Parameters:
#' * `output_size` :: `integer()`\cr
#'   The target output size. Can be a single number or a vector.
#' @templateVar id nn_adaptive_avg_pool2d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchAdaptiveAvgPool2D = R6Class("PipeOpTorchAdaptiveAvgPool2D", inherit = PipeOpTorchAdaptiveAvgPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_adaptive_avg_pool2d", param_vals = list()) {
      super$initialize(id = id, d = 2, param_vals = param_vals)
    }
  )
)

#' @title 3D Adaptive Average Pooling
#'
#' @inherit torch::nnf_adaptive_avg_pool3d description
#'
#' @section nn_module:
#' Calls [`nn_adaptive_avg_pool3d()`][torch::nn_adaptive_avg_pool3d] during training.
#' @section Parameters:
#' * `output_size` :: `integer()`\cr
#'   The target output size. Can be a single number or a vector.
#' @templateVar id nn_adaptive_avg_pool3d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchAdaptiveAvgPool3D = R6Class("PipeOpTorchAdaptiveAvgPool3D", inherit = PipeOpTorchAdaptiveAvgPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_adaptive_avg_pool3d", param_vals = list()) {
      super$initialize(id = id, d = 3, param_vals = param_vals)
    }
  )
)

#' @include aaa.R
register_po("nn_adaptive_avg_pool1d", PipeOpTorchAdaptiveAvgPool1D)
register_po("nn_adaptive_avg_pool2d", PipeOpTorchAdaptiveAvgPool2D)
register_po("nn_adaptive_avg_pool3d", PipeOpTorchAdaptiveAvgPool3D)
