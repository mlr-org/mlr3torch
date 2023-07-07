PipeOpTorchMaxPool = R6Class("PipeOpTorchMaxPool",
  inherit = PipeOpTorch,
  public = list(
    #  @description Creates a new instance of this [R6][R6::R6Class] class.
    #  @template params_pipelines
    #  @param d (`integer(1)`)\cr
    #    The dimension of the max pooling operation.
    #  @param return_indices (`logical(1)`)\cr
    #   Whether to return the indices. See section 'Input and Output Channels' for more information.
    initialize = function(id, d, return_indices = FALSE, param_vals = list()) {
      private$.d = assert_int(d, lower = 1, upper = 3, coerce = TRUE)
      module_generator = switch(private$.d, nn_max_pool1d, nn_max_pool2d, nn_max_pool3d)
      check_vector = make_check_vector(d)
      param_set = ps(
        kernel_size = p_uty(custom_check = check_vector, tags = c("required", "train")),
        padding = p_uty(default = 0L, custom_check = check_vector, tags = "train"),
        stride = p_uty(default = NULL, custom_check = check_vector, tags = "train"),
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
    .shapes_out = function(shapes_in, param_vals, task) {
      res = list(max_output_shape(
        shape_in = shapes_in[[1]],
        conv_dim = private$.d,
        padding = param_vals$padding %??% 0,
        stride = param_vals$stride %??% param_vals$kernel_size,
        kernel_size = param_vals$kernel_size,
        ceil_mode = param_vals$ceil_mode %??% FALSE
      ))

      if (private$.return_indices) rep(res, 2) else res
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals, list(return_indices = private$.return_indices))
    },
    .return_indices = NULL,
    .d = NULL
  )
)

max_output_shape = avg_output_shape

#' @title 1D Max Pooling
#'
#' @templateVar id nn_max_pool1d
#' @section Input and Output Channels:
#' If `return_indices` is `FALSE` during construction, there is one input channel 'input' and one output channel 'output'.
#' If `return_indices` is `TRUE`, there are two output channels 'output' and 'indices'.
#' For an explanation see [`PipeOpTorch`].
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_max_pool1d description
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
#' Calls [`torch::nn_max_pool1d()`] during training.
#' @export
PipeOpTorchMaxPool1D = R6Class("PipeOpTorchMaxPool1D", inherit = PipeOpTorchMaxPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @param return_indices (`logical(1)`)\cr
    #'  Whether to return the indices.
    #'  If this is `TRUE`, there are two output channels `"output"` and `"indices"`.
    initialize = function(id = "nn_max_pool1d", return_indices = FALSE, param_vals = list()) {
      super$initialize(id = id, d = 1, return_indices = return_indices, param_vals = param_vals)
    }
  )
)

#' @title 2D Max Pooling
#'
#' @templateVar id nn_max_pool2d
#' @inheritSection mlr_pipeops_nn_max_pool1d Input and Output Channels
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_max_pool2d description
#'
#' @inheritSection mlr_pipeops_nn_max_pool1d Parameters
#' @section Internals:
#' Calls [`torch::nn_max_pool2d()`] during training.
#' @export
PipeOpTorchMaxPool2D = R6Class("PipeOpTorchMaxPool2D", inherit = PipeOpTorchMaxPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @param return_indices (`logical(1)`)\cr
    #'  Whether to return the indices.
    #'  If this is `TRUE`, there are two output channels `"output"` and `"indices"`.
    initialize = function(id = "nn_max_pool2d", return_indices = FALSE, param_vals = list()) {
      super$initialize(id = id, d = 2, return_indices = return_indices, param_vals = param_vals)
    }
  )
)


#' @title 3D Max Pooling
#'
#' @templateVar id nn_max_pool3d
#' @inheritSection mlr_pipeops_nn_max_pool1d Input and Output Channels
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nnf_max_pool3d description
#'
#' @inheritSection mlr_pipeops_nn_max_pool1d Parameters
#'
#' @section Internals:
#' Calls [`torch::nn_max_pool3d()`] during training.
#'
#' @export
PipeOpTorchMaxPool3D = R6Class("PipeOpTorchMaxPool3D", inherit = PipeOpTorchMaxPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @param return_indices (`logical(1)`)\cr
    #'  Whether to return the indices.
    #'  If this is `TRUE`, there are two output channels `"output"` and `"indices"`.
    initialize = function(id = "nn_max_pool3d", return_indices = FALSE, param_vals = list()) {
      super$initialize(id = id, d = 3, return_indices = return_indices, param_vals = param_vals)
    }
  )
)

#' @include zzz.R
register_po("nn_max_pool1d", PipeOpTorchMaxPool1D)
register_po("nn_max_pool2d", PipeOpTorchMaxPool2D)
register_po("nn_max_pool3d", PipeOpTorchMaxPool3D)
