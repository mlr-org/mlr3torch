#' @title Transpose Convolution
#' @description
#' 1D, 2D and 3D Tranpose Convolution.
#'
#' @section Calls:
#' Calls `nn_conv_transpose1d`, `nn_conv_transpose2d` or `nn_conv_transpose3d`.
#'
#' @section Custom mlr3 parameters:
#' * `in_channels` - This parameter is inferred as the second dimension of the input tensor.
#'
#' @export
NULL

PipeOpTorchConvTranspose = R6Class("PipeOpTorchConvTranspose",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, d, module_generator, return_indices = FALSE, param_vals = list()) {
      private$.d = assert_int(d)

      param_set = ps(
        out_channels = p_int(lower = 1L, tags = c("required", "train")),
        kernel_size = p_uty(custom_check = check_vector(d), tags = c("required", "train")),
        stride = p_uty(default = 1L, custom_check = check_vector(d), tags = "train"),
        padding = p_uty(default = 0L, custom_check = check_vector(d), tags = "train"),
        output_padding = p_uty(default = 0L, custom_check = check_vector(d), tags = "train"),
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
      list(conv_transpose_output_shape(
        shape_in = shapes_in[[1]],
        dim = private$.d,
        padding = param_vals$padding %??% 0,
        dilation = param_vals$dilation %??% 1,
        stride = param_vals$stride %??% 1,
        kernel_size = param_vals$kernel_size,
        output_padding = param_vals$output_padding,
        out_channels = param_vals$out_channels
      ))
    },
    .d = NULL
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname conv_transpose
#' @export
PipeOpTorchConvTranspose1D = R6Class("PipeOpTorchConvTranspose1D", inherit = PipeOpTorchConvTranspose,
  public = list(
    initialize = function(id = "conv1d", param_vals = list()) {
      super$initialize(id = id, d = 1, module_generator = nn_conv1d, param_vals = param_vals)
    }
  )
)


#' @template param_id
#' @template param_param_vals
#' @rdname conv_transpose
#' @export
PipeOpTorchConvTranspose2D = R6Class("PipeOpTorchConvTranspose2D", inherit = PipeOpTorchConvTranspose,
  public = list(
    initialize = function(id = "conv2d", param_vals = list()) {
      super$initialize(id = id, d = 2, module_generator = nn_conv2d, param_vals = param_vals)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname conv_transpose
#' @export
PipeOpTorchConvTranspose3D = R6Class("PipeOpTorchConvTranspose3D", inherit = PipeOpTorchConvTranspose,
  public = list(
    initialize = function(id = "conv3d", param_vals = list()) {
      super$initialize(id = id, d = 3, module_generator = nn_conv3d, param_vals = param_vals)
    }
  )
)

#' @include zzz.R
register_po("nn_conv_transpose1d", PipeOpTorchConvTranspose1D)
register_po("nn_conv_transpose2d", PipeOpTorchConvTranspose2D)
register_po("nn_conv_transpose3d", PipeOpTorchConvTranspose3D)


conv_transpose_output_shape = function(shape_in, dim, padding, dilation, stride, kernel_size, output_padding, out_channels) {
  assert_int(shape_in, min.len = conv_dim)
  shape_head = utils::head(shape_in, -(dim + 1))
  shape_tail = utils::tail(shape_in, dim)
  c(shape_head, out_channels,
    (shape_tail - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
  )
}

