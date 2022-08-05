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
#' @template param_id
#' @template param_param_vals
#' @name conv_transpose
NULL

#' @template param_id
#' @template param_param_vals
#' @rdname conv_transpose
#' @export
TorchOpConvTranspose1D = R6Class("TorchOpConvTranspose1D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "conv_transpose1d", param_vals = list()) {
      param_set = make_paramset_tranpose(1L)
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )

    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tags = "train")
      assert_true(length(inputs$input$shape) == 3L)
      param_vals$in_channels = inputs$input$shape[2L]
      invoke(nn_conv_transpose1d, .args = param_vals)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname conv_transpose
#' @export
TorchOpConvTranspose2D = R6Class("TorchOpConvTranspose2D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "conv_transpose2d", param_vals = list()) {
      param_set = make_paramset_tranpose(2L)
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )

    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tags = "train")
      assert_true(length(inputs$input$shape) == 4L)
      param_vals$in_channels = inputs$input$shape[2L]
      invoke(nn_conv_transpose2d, .args = param_vals)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname conv_transpose
#' @export
TorchOpConvTranspose3D = R6Class("TorchOpConvTranspose3D",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "conv_transpose3d", param_vals = list()) {
      param_set = make_paramset_tranpose(3L)
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )

    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tags = "train")
      assert_true(length(inputs$input$shape) == 5L)
      param_vals$in_channels = inputs$input$shape[2L]
      invoke(nn_conv_transpose2d, .args = param_vals)
    }
  )
)

make_paramset_conv_transpose = function(d) {
  force(d)
  check_fn = function(x) {
    if (is.null(x) || test_integerish(x, any.missing = FALSE) && (length(x) %in% c(1, d))) {
      return(TRUE)
    }
    sprintf("Must be an integerish vector of length 1 or %s", d)
  }
  param_set = ps(
    out_channels = p_int(lower = 1L, tags = c("required", "train")),
    kernel_size = p_uty(custom_check = check_fn, tags = c("required", "train")),
    stride = p_uty(default = 1L, custom_check = check_fn, tags = "train")
  )
}

#' @include mlr_torchops.R
mlr_torchops$add("conv_transpose1d", TorchOpConvTranspose1D)
#' @include mlr_torchops.R
mlr_torchops$add("conv_transpose2d", TorchOpConvTranspose2D)
#' @include mlr_torchops.R
mlr_torchops$add("conv_transpose3d", TorchOpConvTranspose3D)
