#' @title Convolution
#' @description
#' 1D, 2D and 3D Convolution
#'
#' @section Calls:
#' Calls `torch::nn_conv1d()`, `torch::nn_conv2d` or `torch::nn_conv3d()`.
#'
#' @section Custom mlr3 parameters:
#' * `in_channels` - This parameter is inferred as the second dimension of the input tensor.
#'
#' @name conv
NULL

TorchOpConv = R6Class("TorchOpConv",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, d, module_generator, return_indices = FALSE, param_vals = list()) {
      private$.d = assert_int(d)

      param_set = ps(
        out_channels = p_int(lower = 1L, tags = c("required", "train")),
        kernel_size = p_uty(custom_check = check_fn(d), tags = c("required", "train")),
        stride = p_uty(default = 1L, custom_check = check_fn(d), tags = "train"),
        padding = p_uty(default = 0L, custom_check = check_fn(d), tags = "train"),
        dilation = p_uty(default = 1L, custom_check = check_fn(d), tags = "train"),
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
        ceil_mode = FALSE
      ))
    },
    .shape_dependent_params = function(shapes_in) list(),
    .d = NULL
  )
)


#' @template param_id
#' @template param_param_vals
#' @rdname conv
#' @export
TorchOpConv1D = R6Class("TorchOpConv1D", inherit = TorchOpConv,
  public = list(
    initialize = function(id = "conv1d", param_vals = list()) {
      super$initialize(id = id, d = 1, module_generator = nn_conv1d, param_vals = param_vals)
    }
  )
)


#' @template param_id
#' @template param_param_vals
#' @rdname conv
#' @export
TorchOpConv2D = R6Class("TorchOpConv2D", inherit = TorchOpConv,
  public = list(
    initialize = function(id = "conv2d", param_vals = list()) {
      super$initialize(id = id, d = 2, module_generator = nn_conv2d, param_vals = param_vals)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname conv
#' @export
TorchOpConv2D = R6Class("TorchOpConv2D", inherit = TorchOpConv,
  public = list(
    initialize = function(id = "conv2d", param_vals = list()) {
      super$initialize(id = id, d = 2, module_generator = nn_conv2d, param_vals = param_vals)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("conv1d", TorchOpConv1D)
#' @include mlr_torchops.R
mlr_torchops$add("conv2d", TorchOpConv2D)
#' @include mlr_torchops.R
mlr_torchops$add("conv3d", TorchOpConv3D)



conv_output_shape = function(shape_in, conv_dim, padding, dilation, stride, kernel_size, ceil_mode = FALSE) {
  assert_int(shape_in, min.len = conv_dim)
  shape_head = utils::head(shape_in, -conv_dim)
  shape_tail = utils::tail(shape_in, conv_dim)
  c(shape_head,
    (if (ceil_mode) base::ceiling else base::floor)((shape_tail + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
  )
}

check_fn = function(d) function(x) {
  if (is.null(x) || test_integerish(x, any.missing = FALSE) && (length(x) %in% c(1, d))) {
    return(TRUE)
  }
  sprintf("Must be an integerish vector of length 1 or %s", d)
}
