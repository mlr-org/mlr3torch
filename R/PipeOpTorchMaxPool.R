#' @title Base Class for Max Pooling
#'
#' @usage NULL
#' @name mlr_pipeops_torch_max_pool
#' @format `r roxy_pipeop_torch_format()`
#'
#' @description
#' Base class for max pooling.
#' Don't use this class directly.
#'
#' @section Construction:
#' ```
#' PipeOpTorchMaxPool$new(id, d, return_indices = FALSE, param_vals = list())
#' ```
#' `r roxy_param_id()`
#' `r roxy_param_param_vals()`
#' * `d` :: `integer(1)`\cr
#'   The dimension of the max pooling operation.
#' * `return_indices` :: `logical(1)`\cr
#'  Whether to return the indices. See section 'Input and Output Channels' for more information.
#'
#' @section Input and Output Channels:
#' There is one input channel `"input"`.
#' Depending on the constructor argument `return_indices`, there is either one output channel `"output"` if
#' `return_indices` is `FALSE`, or two channels `"output"` and `"indices"` if `return_indices` is `TRUE`.
#' For an explanation see [`PipeOpTorch`].
#'
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `kernel_size` :: `integer()`\cr
#'   The size of the window. Can be single number or a vector.
#' * `stride` :: (`integer(1))`\cr
#'   The stride of the window. Can be a single number or a vector. Default: `kernel_size`
#' * `padding` :: `integer()`\cr
#'  Implicit zero paddings on both sides of the input. Can be a single number or a tuple (padW,). Default: 0
#' * `dilation` :: `integer()`\cr
#'   Controls the spacing between the kernel points; also known as the à trous algorithm. Default: 1
#' * `ceil_mode` :: `logical(1)`\cr
#'   When True, will use ceil instead of floor to compute the output shape. Default: `FALSE`
#'
#' @section Internals:
#' See the respective child class.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchMaxPool = R6Class("PipeOpTorchMaxPool",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, d, return_indices = FALSE, param_vals = list()) {
      private$.d = assert_int(d, lower = 1, upper = 3, coerce = TRUE)
      module_generator = switch(private$.d, nn_max_pool1d, nn_max_pool2d, nn_max_pool3d)
      param_set = ps(
        kernel_size = p_uty(custom_check = check_vector(d), tags = c("required", "train")),
        stride = p_uty(default = NULL, custom_check = check_vector(d), tags = "train"),
        padding = p_uty(default = 0L, custom_check = check_vector(d), tags = "train"),
        dilation = p_int(default = 1L, tags = "train"),
        ceil_mode = p_lgl(default = FALSE, tags = "train")
      )

      private$.return_indices = assert_flag(return_indices)

      super$initialize(
        id = id,
        module_generator = module_generator,
        param_vals = param_vals,
        param_set = param_set,
        outname = if (return_indices) c("output", "indices") else "output",
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
    .shape_dependent_params = function(shapes_in, param_vals) {
      c(param_vals, list(return_indices = private$.return_indices))
    },
    .return_indices = NULL,
    .d = NULL
  )
)

#' @title 1D Max Pooling
#'
#' @usage NULL
#' @name mlr_pipeops_torch_max_pool1d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_max_pool1d description
#'
#' @section Construction:
#' ```
#' PipeOpTorchMaxPool1D$new(id = "nn_max_pool1d", return_indices = FALSE, param_vals = list())
#' ```
#' `r roxy_param_id("nn_max_pool1d")`
#' `r roxy_param_param_vals()`
#' * `return_indices` :: `logical(1)`\cr
#'   Whether to return the indices as well, in which case there are two output channels `"output"` and `"indices"`.
#'
#' @inheritSection mlr_pipeops_torch_max_pool Input and Output Channels
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_max_pool Parameters
#'
#' @section Internals:
#' Calls [`torch::nn_max_pool3d()`] during training.
#' The
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#'
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_max_pool1d", kernel_size = 3)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 100))
#'
#' # pot
#' obj = pot("max_pool1d")
#' obj$id
#'
PipeOpTorchMaxPool1D = R6Class("PipeOpTorchMaxPool1D", inherit = PipeOpTorchMaxPool,
  public = list(
    initialize = function(id = "nn_max_pool1d", return_indices = FALSE, param_vals = list()) {
      super$initialize(id = id, d = 1, return_indices = return_indices, param_vals = param_vals)
    }
  )
)

#' @title 2D Max Pooling
#'
#' @usage NULL
#' @name mlr_pipeops_torch_max_pool2d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_max_pool2d description
#'
#' @section Construction:
#' ```
#' PipeOpTorchMaxPool2D$new(id = "nn_max_pool2d", return_indices = FALSE, param_vals = list())
#' ```
#' `r roxy_param_id("nn_max_pool2d")`
#' `r roxy_param_param_vals()`
#' * `return_indices` :: `logical(1)`\cr
#'   Whether to return the indices as well, in which case there are two output channels `"output"` and `"indices"`.
#'
#' @inheritSection mlr_pipeops_torch_max_pool Input and Output Channels
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_max_pool Parameters
#'
#' @section Internals:
#' Calls [`torch::nn_max_pool2d()`] during training.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_max_pool2d", kernel_size = 3)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 100, 100))
#'
#' # pot
#' obj = pot("max_pool2d")
#' obj$id
#'
PipeOpTorchMaxPool2D = R6Class("PipeOpTorchMaxPool2D", inherit = PipeOpTorchMaxPool,
  public = list(
    initialize = function(id = "nn_max_pool2d", return_indices = FALSE, param_vals = list()) {
      super$initialize(id = id, d = 2, return_indices = return_indices, param_vals = param_vals)
    }
  )
)


#' @title 3D Max Pooling
#'
#' @usage NULL
#' @name mlr_pipeops_torch_max_pool3d
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_max_pool3d description
#'
#' @section Construction:
#' ```
#' PipeOpTorchMaxPool3D$new(id = "nn_max_pool3d", return_indices = FALSE, param_vals = list())
#' ```
#' `r roxy_param_id("nn_max_pool3d")`
#' `r roxy_param_param_vals()`
#' * `return_indices` :: `logical(1)`\cr
#'   Whether to return the indices as well, in which case there are two output channels `"output"` and `"indices"`.
#'
#' @inheritSection mlr_pipeops_torch_max_pool Input and Output Channels
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @inheritSection mlr_pipeops_torch_max_pool Parameters
#'
#' @section Internals:
#' Calls [`torch::nn_max_pool3d()`] during training.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
#' @examples
#' # po
#' obj = po("nn_max_pool3d", kernel_size = 3)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(c(16, 5, 100, 100, 100))
#'
#' # pot
#' obj = pot("max_pool3d")
#' obj$id
PipeOpTorchMaxPool3D = R6Class("PipeOpTorchMaxPool3D", inherit = PipeOpTorchMaxPool,
  public = list(
    initialize = function(id = "nn_max_pool3d", return_indices = FALSE, param_vals = list()) {
      super$initialize(id = id, d = 3, return_indices = return_indices, param_vals = param_vals)
    }
  )
)

#' @include zzz.R
register_po("nn_max_pool1d", PipeOpTorchMaxPool1D)
register_po("nn_max_pool2d", PipeOpTorchMaxPool2D)
register_po("nn_max_pool3d", PipeOpTorchMaxPool3D)
