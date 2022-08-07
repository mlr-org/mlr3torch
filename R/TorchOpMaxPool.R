#' @title Max Pooling
#' @description
#' 1D, 2D and 3D Max Pooling.
#' @section Calls:
#' * `nn_max_pool1d`
#' * `nn_max_pool2d`
#' * `nn_max_pool3d`
#'
#' @name max_pool
NULL

TorchOpMaxPool = R6Class("TorchOpMaxPool",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, d, module_generator, return_indices = FALSE, param_vals = list()) {
      private$.d = assert_int(d)

      param_set = ps(
        kernel_size = p_uty(custom_check = check_fn(d), tags = c("required", "train")),
        stride = p_uty(default = NULL, custom_check = check_fn(d), tags = "train"),
        padding = p_uty(default = 0L, custom_check = check_fn(d), tags = "train"),
        dilation = p_int(default = 1L, tags = "train"),
        ceil_mode = p_lgl(default = FALSE, tags = "train")
      )

      private$.return_indices = assert_flag(return_indices)

      super$initialize(
        id = id,
        module_generator = module_generator,
        param_vals = param_vals,
        param_set = param_set,
        multi_output = if (return_indices) 2,
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      res = list(conv_output_shape(
        shape_in = shapes_in[[1]],
        conv_dim = private$.d,
        padding = param_vals$padding %??% 0,
        dilation = param_vals$dilation %??% 1,
        stride = param_vals$stride %??% param_vals$kernel_size,
        kernel_size = param_vals$kernel_size,
        ceil_mode = param_vals$ceil_mode %??% FALSE
      ))

      if (private$.return_indices) rep(res, 2) else res
    },
    .shape_dependent_params = function(shapes_in) {
      list(return_indices = private$.return_indices)
    },
    .return_indices = NULL,
    .d = NULL
  )
)

#' @param return_indices (`logical(1)`)\cr
#'   Whether to return the inidices.
#' @template param_id
#' @template param_param_vals
#' @rdname max_pool
#' @export
TorchOpMaxPool1D = R6Class("TorchOpMaxPool1D", inherit = TorchOpMaxPool,
  public = list(
    initialize = function(id = "max_pool1d", return_indices = FALSE, param_vals = list()) {
      super$initialize(id = id, d = 1, module_generator = nn_max_pool1d, return_indices = return_indices, param_vals = param_vals)
    }
  )
)

#' @param return_indices (`logical(1)`)\cr
#'   Whether to return the inidices.
#' @template param_id
#' @template param_param_vals
#' @rdname max_pool
#' @export
TorchOpMaxPool2D = R6Class("TorchOpMaxPool2D", inherit = TorchOpMaxPool,
  public = list(
    initialize = function(id = "max_pool2d", return_indices = FALSE, param_vals = list()) {
      super$initialize(id = id, d = 2, module_generator = nn_max_pool2d,  return_indices = return_indices, param_vals = param_vals)
    }
  )
)


#' @param return_indices (`logical(1)`)\cr
#'   Whether to return the inidices.
#' @template param_id
#' @template param_param_vals
#' @rdname max_pool
#' @export
TorchOpMaxPool3D = R6Class("TorchOpMaxPool3D", inherit = TorchOpMaxPool,
  public = list(
    initialize = function(id = "max_pool3d", return_indices = FALSE, param_vals = list()) {
      super$initialize(id = id, d = 3, module_generator = nn_max_pool3d, return_indices = return_indices, param_vals = param_vals)
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("max_pool1d", TorchOpMaxPool1D)
#' @include mlr_torchops.R
mlr_torchops$add("max_pool2d", TorchOpMaxPool2D)
#' @include mlr_torchops.R
mlr_torchops$add("max_pool3d", TorchOpMaxPool3D)
