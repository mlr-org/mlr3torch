PipeOpTorchAvgPool = R6Class("PipeOpTorchAvgPool",
  inherit = PipeOpTorch,
  public = list(
    #  @description Creates a new instance of this [R6][R6::R6Class] class.
    #  @template params_pipelines
    #  @param d (`integer(1)`)\cr
    #    The dimension for the average pooling operation.
    initialize = function(id, d, param_vals = list()) {
      private$.d = assert_int(d, lower = 1, upper = 3)
      module_generator = switch(d, nn_avg_pool1d, nn_avg_pool2d, nn_avg_pool3d)
      check_vector = make_check_vector(d)
      param_set = ps(
        kernel_size = p_uty(custom_check = check_vector, tags = c("required", "train")),
        stride = p_uty(default = NULL, custom_check = check_vector, tags = "train"),
        padding = p_uty(default = 0L, custom_check = check_vector, tags = "train"),
        ceil_mode = p_lgl(default = FALSE, tags = "train"),
        count_include_pad = p_lgl(default = TRUE, tags = "train")
      )
      if (d >= 2L) {
        param_set = c(param_set, ps(
          divisor_override = p_dbl(default = NULL, lower = 0, tags = "train", special_vals = list(NULL))
        ))
      }

      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        module_generator = module_generator
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      list(avg_output_shape(
        shape_in = shapes_in[[1]],
        conv_dim = private$.d,
        padding = param_vals$padding %??% 0,
        stride = param_vals$stride %??% param_vals$kernel_size,
        kernel_size = param_vals$kernel_size,
        ceil_mode = param_vals$ceil_mode %??% FALSE
      ))
    },
    .d = NULL
  )
)

avg_output_shape = function(shape_in, conv_dim, padding, stride, kernel_size, ceil_mode = FALSE) {
  shape_in = assert_integerish(shape_in, min.len = conv_dim, coerce = TRUE)

  if (length(padding) == 1) padding = rep(padding, conv_dim)
  if (length(stride) == 1) stride = rep(stride, conv_dim)
  if (length(kernel_size) == 1) kernel_size = rep(kernel_size, conv_dim)

  shape_head = utils::head(shape_in, -conv_dim)
  shape_tail = utils::tail(shape_in, conv_dim)
  if (length(shape_head) <= 1) warningf("Input tensor does not have batch dimension.")
  shape_tail = (if (ceil_mode) base::ceiling else base::floor)((shape_tail + 2 * padding - kernel_size) / stride + 1)
  c(shape_head, shape_tail)
}

#' @title 1D Average Pooling
#'
#' @templateVar id nn_avg_pool1d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_adaptive_avg_pool1d description
#'
#' @section Parameters:
#' * `kernel_size` :: (`integer()`)\cr
#'   The size of the window. Can be a single number or a vector.
#' * `stride` :: `integer()`\cr
#'   The stride of the window. Can be a single number or a vector. Default: `kernel_size`.
#' * `padding` :: `integer()`\cr
#'   Implicit zero paddings on both sides of the input. Can be a single number or a vector. Default: 0.
#' * `ceil_mode` :: `integer()`\cr
#'   When `TRUE`, will use ceil instead of floor to compute the output shape. Default: `FALSE`.
#' * `count_include_pad` :: `logical(1)`\cr
#'   When `TRUE`, will include the zero-padding in the averaging calculation. Default: `TRUE`.
#' * `divisor_override` :: `logical(1)`\cr
#'   If specified, it will be used as divisor, otherwise size of the pooling region will be used. Default: NULL.
#'   Only available for dimension greater than 1.
#'
#' @section Internals:
#' Calls [`nn_avg_pool1d()`][torch::nn_avg_pool1d] during training.
#' @export
PipeOpTorchAvgPool1D = R6Class("PipeOpTorchAvgPool1D", inherit = PipeOpTorchAvgPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_avg_pool1d", param_vals = list()) {
      super$initialize(id = id, d = 1, param_vals = param_vals)
    }
  )
)


#' @title 2D Average Pooling
#'
#' @templateVar id nn_avg_pool2d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_adaptive_avg_pool2d description
#'
#' @inheritSection mlr_pipeops_nn_avg_pool1d Parameters
#'
#' @section Internals:
#' Calls [`nn_avg_pool2d()`][torch::nn_avg_pool2d] during training.
#' @export
PipeOpTorchAvgPool2D = R6Class("PipeOpTorchAvgPool2D", inherit = PipeOpTorchAvgPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_avg_pool2d", param_vals = list()) {
      super$initialize(id = id, d = 2, param_vals = param_vals)
    }
  )
)

#' @title 3D Average Pooling
#'
#' @templateVar id nn_avg_pool3d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_adaptive_avg_pool3d description
#'
#' @inheritSection mlr_pipeops_nn_avg_pool1d Parameters
#'
#' @section Internals:
#' Calls [`nn_avg_pool3d()`][torch::nn_avg_pool3d] during training.
#' @export
PipeOpTorchAvgPool3D = R6Class("PipeOpTorchAvgPool3D", inherit = PipeOpTorchAvgPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_avg_pool3d", param_vals = list()) {
      super$initialize(id = id, d = 3, param_vals = param_vals)
    }
  )
)

#' @include zzz.R
register_po("nn_avg_pool1d", PipeOpTorchAvgPool1D)
register_po("nn_avg_pool2d", PipeOpTorchAvgPool2D)
register_po("nn_avg_pool3d", PipeOpTorchAvgPool3D)
