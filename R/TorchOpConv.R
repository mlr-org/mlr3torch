#' @title Convolution
#' @description
#' 1D, 2D and 3D Convolution. The number of input channels is inferred from the second tensor
#' dimension.
#' @section Calls:
#'  * `"conv1d"`:
#' @name conv
NULL

#' @template param_id
#' @template param_param_vals
#' @rdname conv
#' @export
TorchOpConv1D = R6Class("TorchOpConv1D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "conv1d", param_vals = list()) {
      param_set = make_paramset_conv(d = 1L)
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      input = inputs$input
      assert_true(length(input$shape) == 3L)
      args = insert_named(
        param_vals,
        list(in_channels = input$shape[2L])
      )
      invoke(nn_conv1d, .args = args)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname conv
#' @export
TorchOpConv2D = R6Class("TorchOpConv2D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "conv2d", param_vals = list()) {
      param_set = make_paramset_conv(d = 2L)
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      input = inputs$input
      assert_true(length(input$shape) == 4L)
      args = insert_named(
        param_vals,
        list(in_channels = input$shape[2L])
      )
      invoke(nn_conv2d, .args = args)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname conv
#' @export
TorchOpConv3D = R6Class("TorchOpConv3D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "conv3d", param_vals = list()) {
      param_set = make_paramset_conv(d = 3L)
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      input = inputs$input
      assert_true(length(input$shape) == 5L)
      args = insert_named(
        param_vals,
        list(in_channels = input$shape[2L])
      )
      invoke(nn_conv3d, .args = args)
    }
  )
)

make_paramset_conv = function(d) {
  force(d)
  check_fn = function(x) {
    check_integerish(x, min.len = 1L, max.len = d, any.missing = FALSE)
  }

  padding_levels = c("zeros", "circular", "replicate")
  if (d != 5L) {
    padding_levels[4L] = "reflect"
  }

  param_set = ps(
    out_channels = p_int(lower = 1L, tags = c("required", "train")),
    kernel_size = p_uty(custom_check = check_fn, tags = c("required", "train")),
    stride = p_uty(default = 1L, custom_check = check_fn, tags = "train"),
    padding = p_uty(default = 0L, custom_check = check_fn, tags = "train"),
    dilation = p_uty(default = 1L, custom_check = check_fn, tags = "train"),
    groups = p_int(default = 1L, lower = 1L, tags = "train"),
    bias = p_lgl(default = TRUE, tags = "train"),
    padding_mode = p_fct(
      default = "zeros",
      levels = padding_levels,
      tags = "train"
    )
  )
}

#' @include mlr_torchops.R
mlr_torchops$add("conv1d", TorchOpConv1D)
#' @include mlr_torchops.R
mlr_torchops$add("conv2d", TorchOpConv2D)
#' @include mlr_torchops.R
mlr_torchops$add("conv3d", TorchOpConv3D)
