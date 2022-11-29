#' @title Base Class for Convolution
#'
#' @usage NULL
#' @name mlr_pipeops_torch_conv
#' @format `r roxy_pipeop_torch_format()`
#'
#' @description
#' Base class for transpose convolution.
#' Don't use this class directly.
#'
#' @section Construction: `r roxy_pipeop_torch_construction("Conv")`
#' `r roxy_param_id()`
#' `r roxy_param_param_vals()`
#' `r roxy_param_module_generator()`
#' * `d` :: `integer(1)`\cr
#'   The dimension of the transpose convolution.
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `out_channels` :: `integer(1)`\cr
#'   Number of channels produced by the convolution.
#' * `kernel_size` :: `integer()`\cr
#'   Size of the convolving kernel
#' * `stride` :: `integer()`\cr
#'   Stride of the convolution. The default is 1.
#' * `padding` :: ` (`integer()`)\cr
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
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' See the respective child class.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchConv = R6Class("PipeOpTorchConv",
  inherit = PipeOpTorch,
  public = list(
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
#' @name mlr_pipeops_torch_conv1d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_conv1d description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("Conv1D")`
#' `r roxy_param_id("nn_conv1d")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_conv Parameters
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`torch::nn_conv1d()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_conv1d", out_channels = 4, kernel_size = 3)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 3, 64))
#'
#' # pot
#' obj = po("conv1d")
#' obj$id
PipeOpTorchConv1D = R6Class("PipeOpTorchConv1D", inherit = PipeOpTorchConv,
  public = list(
    initialize = function(id = "nn_conv1d", param_vals = list()) {
      super$initialize(id = id, d = 1, module_generator = nn_conv1d, param_vals = param_vals)
    }
  )
)


#' @title 2D Convolution
#'
#' @usage NULL
#' @name mlr_pipeops_torch_conv2d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_conv2d description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("Conv2D")`
#' `r roxy_param_id("nn_conv2d")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_conv Parameters
#'
#' @section Fields `r roxy_pipeop_torch_fields_default()`
#' @section Methods `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`torch::nn_conv2d()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_conv2d", out_channels = 4, kernel_size = 3)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 3, 64, 64))
#'
#' # pot
#' obj = po("conv2d")
#' obj$id
#'
PipeOpTorchConv2D = R6Class("PipeOpTorchConv2D", inherit = PipeOpTorchConv,
  public = list(
    initialize = function(id = "nn_conv2d", param_vals = list()) {
      super$initialize(id = id, d = 2, module_generator = nn_conv2d, param_vals = param_vals)
    }
  )
)

#' @title 3D Convolution
#'
#' @usage NULL
#' @name mlr_pipeops_torch_conv3d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_conv3d description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("Conv3D")`
#' `r roxy_param_id("nn_conv3d")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_conv Parameters
#'
#' @section Fields `r roxy_pipeop_torch_fields_default()`
#' @section Methods `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`torch::nn_conv3d()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_conv3d", out_channels = 4, kernel_size = 3)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 3, 64, 64, 100))
#'
#' # pot
#' obj = po("conv3d")
#' obj$id
#'
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
  shape_in = assert_integerish(shape_in, min.len = conv_dim, coerce = TRUE)
  shape_head = utils::head(shape_in, -(conv_dim + length(out_channels)))
  shape_tail = utils::tail(shape_in, conv_dim)
  c(shape_head, out_channels,
    (if (ceil_mode) base::ceiling else base::floor)((shape_tail + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
  )
}

