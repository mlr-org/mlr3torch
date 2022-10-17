#' @title
#' @description
#' 1D, 2D and 3D Convolution
#'
#' @section Calls:
#' Calls `torch::nn_conv1d()`, `torch::nn_conv2d` or `torch::nn_conv3d()`.
#'
#' @section Inferred parameters
#' * `in_channels` - This parameter is inferred as the second dimension of the input tensor.
PipeOpTorchConv = R6Class("PipeOpTorchConv",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id, d, module_generator, param_vals = list()) {
      private$.d = assert_int(d)

      param_set = ps(
        out_channels = p_int(lower = 1L, tags = c("required", "train")),
        kernel_size = p_uty(custom_check = check_vector(d), tags = c("required", "train")),
        stride = p_uty(default = 1L, custom_check = check_vector(d), tags = "train"),
        padding = p_uty(default = 0L, custom_check = check_vector(d), tags = "train"),
        dilation = p_uty(default = 1L, custom_check = check_vector(d), tags = "train"),
        groups = p_int(default = 1L, lower = 1L, tags = "train"),
        bias = p_lgl(default = TRUE, tags = "train"),
        padding_mode = p_fct(
          default = "zeros",
          levels = c("zeros", "circular", "replicate", "reflect"),
          tags = "train"
        )
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
    .shapes_out = function(shapes_in, param_vals) {
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
    .shape_dependent_params = function(shapes_in, param_vals) {
      c(param_vals, in_channels = shapes_in[1])

    },
    .d = NULL
  )
)


#' @title 1D Convolution
#'
#' @usage NULL
#' @template pipeop_torch_format
#'
#' @inherit torch::nn_conv1d description
#'
#' @section Torch Module:
#' Wraps torch module [`torch::nn_conv1d`].
#'
#' @template pipeop_torch_channels
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#'
#' * The parameter `in_features` is automaticll
#'
#' * `out_channels` :: (`integer(1))\cr
#'   Number of channels produced by the convolution
#'
#' * `kernel_size` :: (`integer()`)\cr
#'   Size of the convolving kernel
#'
#' * `stride` :: (`integer()`)\cr
#'   Stride of the convolution. Default:
#' * `padding` :: (`integer()`)\cr
#'   Padding added to both sides of the input. Default: 0
#' * `dilation` :: (`integer()`)\cr
#'   Spacing between kernel elements.
#' * `groups` :: (int, optional): Number of blocked connections from input
#'           channels to output channels. Default: 1
#'
#'     bias: (bool, optional): If ‘TRUE’, adds a learnable bias to the
#'           output. Default: ‘TRUE’
#'
#' padding_mode: (string, optional): ‘'zeros'’, ‘'reflect'’, ‘'replicate'’
#'           or ‘'circular'’. Default: ‘'zeros'’
#'
#'
#'
#' @template param_id
#' @template param_param_vals
#' @rdname conv
#' @export
PipeOpTorchConv1D = R6Class("PipeOpTorchConv1D", inherit = PipeOpTorchConv,
  public = list(
    initialize = function(id = "nn_conv1d", param_vals = list()) {
      super$initialize(id = id, d = 1, module_generator = nn_conv1d, param_vals = param_vals)
    }
  )
)


#' @template param_id
#' @template param_param_vals
#' @rdname conv
#' @export
PipeOpTorchConv2D = R6Class("PipeOpTorchConv2D", inherit = PipeOpTorchConv,
  public = list(
    initialize = function(id = "nn_conv2d", param_vals = list()) {
      super$initialize(id = id, d = 2, module_generator = nn_conv2d, param_vals = param_vals)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname conv
#' @export
PipeOpTorchConv3D = R6Class("PipeOpTorchConv3D", inherit = PipeOpTorchConv,
  public = list(
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
  assert_int(shape_in, min.len = conv_dim)
  shape_head = utils::head(shape_in, -(conv_dim + length(out_channels)))
  shape_tail = utils::tail(shape_in, conv_dim)
  c(shape_head, out_channels,
    (if (ceil_mode) base::ceiling else base::floor)((shape_tail + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
  )
}

