PipeOpTorchConv = R6Class("PipeOpTorchConv",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id, d, module_generator, param_vals = list()) {
      private$.d = assert_int(d)

      check_vector = make_check_vector(d)
      param_set = ps(
        out_channels = p_int(lower = 1L, tags = c("required", "train")),
        kernel_size = p_uty(custom_check = check_vector, tags = c("required", "train")),
        stride = p_uty(default = 1L, custom_check = check_vector, tags = "train"),
        padding = p_uty(default = 0L, custom_check = check_vector, tags = "train"),
        dilation = p_uty(default = 1L, custom_check = check_vector, tags = "train"),
        groups = p_int(default = 1L, lower = 1L, tags = "train"),
        bias = p_lgl(default = TRUE, tags = "train"),
        padding_mode = p_fct(default = "zeros", levels = c("zeros", "circular", "replicate", "reflect"),
          tags = "train")
      )

      super$initialize(
        id = id,
        module_generator = module_generator,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      list(conv_output_shape(
        shape_in = shapes_in[[1]],
        conv_dim = private$.d,
        padding = param_vals$padding %??% 0,
        dilation = param_vals$dilation %??% 1,
        stride = param_vals$stride %??% 1,
        kernel_size = param_vals$kernel_size,
        out_channels = param_vals$out_channels,
        ceil_mode = FALSE
      ))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals, in_channels = unname(shapes_in[[1L]][2L]))

    },
    .d = NULL
  )
)


#' @title 1D Convolution
#'
#' @templateVar id nn_conv1d
#' @template pipeop_torch_channels_default
#' @templateVar param_vals kernel_size = 10, out_channels = 1
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_conv1d description
#'
#' @section Parameters:
#' * `out_channels` :: `integer(1)`\cr
#'   Number of channels produced by the convolution.
#' * `kernel_size` :: `integer()`\cr
#'   Size of the convolving kernel.
#' * `stride` :: `integer()`\cr
#'   Stride of the convolution. The default is 1.
#' * `padding` :: `integer()`\cr
#'  ‘dilation * (kernel_size - 1) - padding’ zero-padding will be added to both sides of the input. Default: 0.
#' * `groups` :: `integer()`\cr
#'   Number of blocked connections from input channels to output channels. Default: 1
#' * `bias` :: `logical(1)`\cr
#'   If ‘TRUE’, adds a learnable bias to the output. Default: ‘TRUE’.
#' * `dilation` :: `integer()`\cr
#'   Spacing between kernel elements. Default: 1.
#' * `padding_mode` :: `character(1)`\cr
#'   The padding mode. One of `"zeros"`, `"reflect"`, `"replicate"`, or `"circular"`. Default is `"zeros"`.
#'
#'
#' @section Internals:
#' Calls [`torch::nn_conv1d()`] when trained.
#' The paramter `in_channels` is inferred from the second dimension of the input tensor.
#' @export
PipeOpTorchConv1D = R6Class("PipeOpTorchConv1D", inherit = PipeOpTorchConv,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_conv1d", param_vals = list()) {
      super$initialize(id = id, d = 1, module_generator = nn_conv1d, param_vals = param_vals)
    }
  )
)


#' @title 2D Convolution
#'
#' @templateVar id nn_conv2d
#' @template pipeop_torch_channels_default
#' @templateVar param_vals kernel_size = 10, out_channels = 1
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_conv2d description
#'
#' @inheritSection mlr_pipeops_nn_conv1d Parameters
#'
#' @section Internals:
#' Calls [`torch::nn_conv2d()`] when trained.
#' The paramter `in_channels` is inferred from the second dimension of the input tensor.
#' @export
PipeOpTorchConv2D = R6Class("PipeOpTorchConv2D", inherit = PipeOpTorchConv,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_conv2d", param_vals = list()) {
      super$initialize(id = id, d = 2, module_generator = nn_conv2d, param_vals = param_vals)
    }
  )
)

#' @title 3D Convolution
#'
#' @templateVar id nn_conv3d
#' @template pipeop_torch_channels_default
#' @templateVar param_vals kernel_size = 10, out_channels = 1
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_conv3d description
#'
#' @inheritSection mlr_pipeops_nn_conv1d Parameters
#'
#' @section Internals:
#' Calls [`torch::nn_conv3d()`] when trained.
#' The paramter `in_channels` is inferred from the second dimension of the input tensor.
#' @export
PipeOpTorchConv3D = R6Class("PipeOpTorchConv3D", inherit = PipeOpTorchConv,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_conv3d", param_vals = list()) {
      super$initialize(id = id, d = 3, module_generator = nn_conv3d, param_vals = param_vals)
    }
  )
)

#' @include zzz.R
register_po("nn_conv1d", PipeOpTorchConv1D)
register_po("nn_conv2d", PipeOpTorchConv2D)
register_po("nn_conv3d", PipeOpTorchConv3D)



conv_output_shape = function(shape_in, conv_dim, padding, dilation, stride, kernel_size, out_channels = NULL, ceil_mode = FALSE) {
  shape_in = assert_integerish(shape_in, min.len = conv_dim + 1, coerce = TRUE)
  shape_head = utils::head(shape_in, -(conv_dim + 1))
  if (length(shape_head) == 0) {
    warningf("Input tensor does not have have batch dimension.")
  }
  shape_tail = utils::tail(shape_in, conv_dim)
  c(shape_head, out_channels,
    (if (ceil_mode) base::ceiling else base::floor)((shape_tail + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
  )
}

