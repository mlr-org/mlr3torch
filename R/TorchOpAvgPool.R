#' @title Average Pooling
#' @description
#' 1D, 2D and 3D Average Pooling.
#' @section Calls:
#' * `nn_avg_pool1d`
#' * `nn_avg_pool2d`
#' * `nn_avg_pool3d`
#'
#' @name avg_pool
NULL

PipeOpTorchAvgPool = R6Class("PipeOpTorchAvgPool",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, d, module_generator, param_vals = list()) {
      private$.d = assert_int(d)

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
        module_generator = module_generator,
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
  assert_int(shape_in, min.len = conv_dim)
  shape_head = utils::head(shape_in, -conv_dim)
  shape_tail = utils::tail(shape_in, conv_dim)
  c(shape_head,
    (if (ceil_mode) base::ceiling else base::floor)((shape_tail + 2 * padding - kernel_size) / stride + 1)
  )
}

#' @template param_id
#' @template param_param_vals
#' @rdname avg_pool
#' @export
PipeOpTorchAvgPool1D = R6Class("PipeOpTorchAvgPool1D", inherit = PipeOpTorchAvgPool,
  public = list(
    initialize = function(id = "nn_avg_pool1d", param_vals = list()) {
      super$initialize(id = id, d = 1, module_generator = nn_avg_pool1d, param_vals = param_vals)
    }
  )
)


#' @template param_id
#' @template param_param_vals
#' @rdname avg_pool
#' @export
PipeOpTorchAvgPool2D = R6Class("PipeOpTorchAvgPool2D", inherit = PipeOpTorchAvgPool,
  public = list(
    initialize = function(id = "nn_avg_pool2d", param_vals = list()) {
      super$initialize(id = id, d = 2, module_generator = nn_avg_pool2d, param_vals = param_vals)
    }
  )
)

#' @template param_id
#' @template param_param_vals
#' @rdname avg_pool
#' @export
PipeOpTorchAvgPool2D = R6Class("PipeOpTorchAvgPool2D", inherit = PipeOpTorchAvgPool,
  public = list(
    initialize = function(id = "nn_avg_pool2d", param_vals = list()) {
      super$initialize(id = id, d = 2, module_generator = nn_avg_pool2d, param_vals = param_vals)
    }
  )
)

#' @include zzz.R
register_po("nn_avg_pool1d", PipeOpTorchAvgPool1D)
register_po("nn_avg_pool2d", PipeOpTorchAvgPool2D)
register_po("nn_avg_pool3d", PipeOpTorchAvgPool3D)
