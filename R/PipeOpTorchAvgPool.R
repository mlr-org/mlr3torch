#' @title Base Class for Average Pooling
#'
#' @usage NULL
#' @name mlr_pipeops_torch_avg_pool
#' @format `r roxy_pipeop_torch_format()`
#'
#' @description
#' Base class for average pooling.
#' Don't use this class directly.
#'
#' @section Construction: `r roxy_pipeop_torch_construction("AvgPool")`
#' `r roxy_param_id()`
#' `r roxy_param_param_vals()`
#' * `d` :: `integer(1)`\cr
#'   The dimension for the average pooling operation.
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
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
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' See the respective child class.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchAvgPool = R6Class("PipeOpTorchAvgPool",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id, d, param_vals = list()) {
      private$.d = assert_int(d, lower = 1, upper = 3)
      module_generator = switch(d, nn_avg_pool1d, nn_avg_pool2d, nn_avg_pool3d)

      param_set = ps(
        kernel_size = p_uty(custom_check = check_vector(d), tags = c("required", "train")),
        stride = p_uty(default = NULL, custom_check = check_vector(d), tags = "train"),
        padding = p_uty(default = 0L, custom_check = check_vector(d), tags = "train"),
        ceil_mode = p_lgl(default = FALSE, tags = "train"),
        count_include_pad = p_lgl(default = TRUE, tags = "train")
      )
      if (d >= 2L) {
        param_set$add(ParamDbl$new("divisor_override", lower = 0, tags = "train"))
      }

      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      list(avg_output_shape(
        shape_in = shapes_in[[1]],
        conv_dim = private$.d,
        padding = param_vals$padding %??% 0,
        stride = param_vals$stride %??% 1,
        kernel_size = param_vals$kernel_size,
        ceil_mode = param_vals$ceil_mode %??% FALSE
      ))
    },
    .d = NULL
  )
)

avg_output_shape = function(shape_in, conv_dim, padding, stride, kernel_size, ceil_mode = FALSE) {
  shape_in = assert_integerish(shape_in, min.len = conv_dim, coerce = TRUE)
  shape_head = utils::head(shape_in, -conv_dim)
  shape_tail = utils::tail(shape_in, conv_dim)
  c(shape_head,
    (if (ceil_mode) base::ceiling else base::floor)((shape_tail + 2 * padding - kernel_size) / stride + 1)
  )
}

#' @title 1D Average Pooling
#'
#' @usage NULL
#' @name mlr_pipeops_torch_avg_pool1d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_adaptive_avg_pool1d description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("AvgPool1D")`
#' `r roxy_param_id("nn_avg_pool1d")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_avg_pool Parameters
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`nn_avg_pool1d()`] during training.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_avg_pool1d", kernel_size = 3)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 100))
#'
#' # pot
#' obj = pot("avg_pool1d")
#' obj$id
#'
PipeOpTorchAvgPool1D = R6Class("PipeOpTorchAvgPool1D", inherit = PipeOpTorchAvgPool,
  public = list(
    initialize = function(id = "nn_avg_pool1d", param_vals = list()) {
      super$initialize(id = id, d = 1, param_vals = param_vals)
    }
  )
)


#' @title 2D Average Pooling
#'
#' @usage NULL
#' @name mlr_pipeops_torch_avg_pool2d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_adaptive_avg_pool2d description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("AvgPool2D")`
#' `r roxy_param_id("nn_avg_pool2d")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_avg_pool Parameters
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`nn_avg_pool2d()`] during training.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_avg_pool2d", kernel_size = 3)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 100))
#'
#' # pot
#' obj = pot("avg_pool2d")
#' obj$id
#'
#' @export
PipeOpTorchAvgPool2D = R6Class("PipeOpTorchAvgPool2D", inherit = PipeOpTorchAvgPool,
  public = list(
    initialize = function(id = "nn_avg_pool2d", param_vals = list()) {
      super$initialize(id = id, d = 2, param_vals = param_vals)
    }
  )
)

#' @title 3D Average Pooling
#'
#' @usage NULL
#' @name mlr_pipeops_torch_avg_pool3d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_adaptive_avg_pool3d description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("AvgPool3D")`
#' `r roxy_param_id("nn_avg_pool3d")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_avg_pool Parameters
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`nn_avg_pool3d()`] during training.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_avg_pool3d", kernel_size = 3)
#' obj$id

#' obj$module_generator
#' obj$shapes_out(c(16, 5, 100))
#'
#' # pot
#' obj = pot("avg_pool3d")
#' obj$id
PipeOpTorchAvgPool3D = R6Class("PipeOpTorchAvgPool3D", inherit = PipeOpTorchAvgPool,
  public = list(
    initialize = function(id = "nn_avg_pool3d", param_vals = list()) {
      super$initialize(id = id, d = 3, param_vals = param_vals)
    }
  )
)

#' @include zzz.R
register_po("nn_avg_pool1d", PipeOpTorchAvgPool1D)
register_po("nn_avg_pool2d", PipeOpTorchAvgPool2D)
register_po("nn_avg_pool3d", PipeOpTorchAvgPool3D)
