#' @title Base Class Transpose Convolution
#'
#' @usage NULL
#' @name mlr_pipeops_torch_conv_transpose
#' @format `r roxy_pipeop_torch_format()`
#'
#' @description
#' Base class for transpose convolution.
#' Don't use this class directly.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchConvTranspose)`
#' * `r roxy_param_id()`
#' * `r roxy_param_param_vals()`
#' * `r roxy_param_module_generator()`
#' * `d` :: `integer(1)`\cr
#'   The dimension of the transpose convolution.
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `out_channels` :: `integer(1)`\cr
#'   Number of output channels produce by the convolution.
#' * `kernel_size` :: `integer()`\cr
#'   Size of the convolving kernel.
#' * `stride` :: `integer()`\cr
#'   Stride of the convolution. Default: 1.
#' * `padding` :: ` `integer()`\cr
#'  ‘dilation * (kernel_size - 1) - padding’ zero-padding will be added to both sides of the input. Default: 0.
#' * `output_padding` ::`integer()`\cr
#'   Additional size added to one side of the output shape. Default: 0.
#' * `groups` :: `integer()`\cr
#'   Number of blocked connections from input channels to output channels. Default: 1
#' * `bias` :: `logical(1)`\cr
#'   If ‘True’, adds a learnable bias to the output. Default: ‘TRUE’.
#' * `dilation` :: `integer()`\cr
#'   Spacing between kernel elements. Default: 1.
#' * `padding_mode` :: `character(1)`\cr
#'   The padding mode. One of `"zeros"`, `"reflect"`, `"replicate"`, or `"circular"`. Default is `"zeros"`.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: See the respective child class.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchConvTranspose = R6Class("PipeOpTorchConvTranspose",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id, d, module_generator, param_vals = list()) {
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
    .shapes_out = function(shapes_in, param_vals, task) {
      list(conv_transpose_output_shape(
        shape_in = shapes_in[[1]],
        dim = private$.d,
        padding = param_vals$padding %??% 0,
        dilation = param_vals$dilation %??% 1,
        stride = param_vals$stride %??% 1,
        kernel_size = param_vals$kernel_size,
        output_padding = param_vals$output_padding %??% 0,
        out_channels = param_vals$out_channels
      ))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals, in_channels = unname(shapes_in[[1L]][2L]))

    },
    .d = NULL
  )
)

#' @title Transpose 1D Convolution
#'
#' @usage NULL
#' @name mlr_pipeops_torch_conv_transpose1d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_conv_transpose1d description
#'
#' @section Construction: `r roxy_construction(PipeOpTorchConvTranspose1D)`
#' * `r roxy_param_id("nn_conv_transpose1d")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#'
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_conv_transpose Parameters
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`nn_conv_transpose1d`].
#' The parameter `in_channels` is inferred as the second dimension of the input tensor.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' obj = po("nn_conv_transpose1d", out_channels = 4, kernel_size = 2)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 3, 64))
PipeOpTorchConvTranspose1D = R6Class("PipeOpTorchConvTranspose1D", inherit = PipeOpTorchConvTranspose,
  public = list(
    initialize = function(id = "nn_conv_transpose1d", param_vals = list()) {
      super$initialize(id = id, d = 1, module_generator = nn_conv_transpose1d, param_vals = param_vals)
    }
  )
)


#' @title Transpose 2D Convolution
#'
#' @usage NULL
#' @name mlr_pipeops_torch_conv_transpose2d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_conv_transpose2d description
#'
#' @section Construction: `r roxy_construction(PipeOpTorchConvTranspose2D)`
#' * `r roxy_param_id("nn_conv_transpose2d")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#'
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_conv_transpose Parameters
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`nn_conv_transpose2d`].
#' The parameter `in_channels` is inferred as the second dimension of the input tensor.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' obj = po("nn_conv_transpose2d", out_channels = 4, kernel_size = 2)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 3, 64, 64))
PipeOpTorchConvTranspose2D = R6Class("PipeOpTorchConvTranspose2D", inherit = PipeOpTorchConvTranspose,
  public = list(
    initialize = function(id = "nn_conv_transpose2d", param_vals = list()) {
      super$initialize(id = id, d = 2, module_generator = nn_conv_transpose2d, param_vals = param_vals)
    }
  )
)

#' @title Transpose 3D Convolution
#'
#' @usage NULL
#' @name mlr_pipeops_torch_conv_transpose3d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_conv_transpose3d description
#'
#' @section Construction: `r roxy_construction(PipeOpTorchConvTranspose3D)`
#' * `r roxy_param_id("nn_conv_transpose3d")`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#'
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_conv_transpose Parameters
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`nn_conv_transpose3d`].
#' The parameter `in_channels` is inferred as the second dimension of the input tensor.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' obj = po("nn_conv_transpose3d", out_channels = 4, kernel_size = 2)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 3, 64, 64, 1000))
PipeOpTorchConvTranspose3D = R6Class("PipeOpTorchConvTranspose3D", inherit = PipeOpTorchConvTranspose,
  public = list(
    initialize = function(id = "nn_conv_transpose3d", param_vals = list()) {
      super$initialize(id = id, d = 3, module_generator = nn_conv_transpose3d, param_vals = param_vals)
    }
  )
)

#' @include zzz.R
register_po("nn_conv_transpose1d", PipeOpTorchConvTranspose1D)
register_po("nn_conv_transpose2d", PipeOpTorchConvTranspose2D)
register_po("nn_conv_transpose3d", PipeOpTorchConvTranspose3D)

conv_transpose_output_shape = function(shape_in, dim, padding, dilation, stride, kernel_size, output_padding,
  out_channels) {
  assert_integerish(shape_in, min.len = dim + 1L)

  if (length(shape_in) ==  dim + 1L) {
    warningf("The input tensor has no batch dimension")
    batch_dimension = integer(0)
  } else {
    batch_dimension = shape_in[1L]
  }

  if (length(padding) == 1) padding = rep(padding, dim)
  if (length(dilation) == 1) dilation = rep(dilation, dim)
  if (length(stride) == 1) stride = rep(stride, dim)
  if (length(kernel_size) == 1) kernel_size = rep(kernel_size, dim)
  if (length(output_padding) == 1) output_padding = rep(output_padding, dim)

  shape_tail = utils::tail(shape_in, dim)
  c(batch_dimension, out_channels,
    (shape_tail - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
  )
}

