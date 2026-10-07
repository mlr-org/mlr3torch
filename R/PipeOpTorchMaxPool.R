PipeOpTorchMaxPool = R6Class("PipeOpTorchMaxPool",
  inherit = PipeOpTorch,
  public = list(
    #  @description Creates a new instance of this [R6][R6::R6Class] class.
    #  @template params_pipelines
    #  @param d (`integer(1)`)\cr
    #    The dimension of the max pooling operation.
    initialize = function(id, d, param_vals = list()) {
      private$.d = assert_int(d, lower = 1, upper = 3, coerce = TRUE)
      module_generator = switch(private$.d, nn_max_pool1d, nn_max_pool2d, nn_max_pool3d)
      check_vector = make_check_vector(d)
      param_set = ps(
        kernel_size = p_uty(custom_check = make_check_vector(d, null_ok = FALSE), tags = c("required", "train")),
        padding = p_uty(default = 0L, custom_check = check_vector, tags = "train"),
        stride = p_uty(default = NULL, custom_check = check_vector, tags = "train"),
        dilation = p_uty(default = 1L, custom_check = check_vector, tags = "train"),
        ceil_mode = p_lgl(default = FALSE, tags = "train")
      )

      super$initialize(
        id = id,
        module_generator = module_generator,
        param_vals = param_vals,
        param_set = param_set,
        outname = c("output", "indices")
      )
      # the indices are only an output channel when they are requested
      self$outputs = "output"
    }
  ),
  private = list(
    .additional_phash_input = function() {
      c(super$.additional_phash_input(), list(d = private$.d))
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      # a pooling operator over `d` dimensions expects `(batch, channels, <d spatial dimensions>)`.
      assert_ndim(shapes_in[[1L]], private$.d + 2L, self$id)
      res = list(pool_output_shape(
        shape_in = shapes_in[[1]],
        conv_dim = private$.d,
        padding = param_vals[["padding"]] %??% 0,
        stride = param_vals[["stride"]] %??% param_vals[["kernel_size"]],
        kernel_size = param_vals[["kernel_size"]],
        dilation = param_vals$dilation %??% 1,
        ceil_mode = param_vals[["ceil_mode"]] %??% FALSE,
        id = self$id
      ))

      rep(res, 2)
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      # the indices are only computed when they are among `$outputs`
      c(param_vals, list(return_indices = "indices" %in% self$output$name))
    },
    # torch returns the indices together with the output, or the output alone
    .module_outputs = function() {
      if ("indices" %in% self$output$name) c("output", "indices") else "output"
    },
    .d = NULL
  )
)

max_output_shape = pool_output_shape

#' @title 1D Max Pooling
#' @inherit torch::nnf_max_pool1d description
#' @section nn_module:
#' Calls [`torch::nn_max_pool1d()`] during training.
#' @section Parameters:
#' * `kernel_size` :: `integer()`\cr
#'   The size of the window. Can be single number or a vector.
#' * `stride` :: `integer()`\cr
#'   The stride of the window. Can be a single number or a vector. Default: `kernel_size`
#' * `padding` :: `integer()`\cr
#'  Implicit zero paddings on both sides of the input. Can be a single number or a tuple (padW,). Default: 0
#' * `dilation` :: `integer()`\cr
#'   Controls the spacing between the kernel points; also known as the a trous algorithm. Default: 1
#' * `ceil_mode` :: `logical(1)`\cr
#'   When True, will use ceil instead of floor to compute the output shape. Default: `FALSE`
#'
#' @templateVar id nn_max_pool1d
#' @section Input and Output Channels:
#' There is one input channel `"input"`.
#' The module has two outputs, `"output"` and `"indices"`, of which only `"output"` is an output
#' channel by default. Set `$outputs` to also (or only) get the indices, e.g.
#' `outputs = c("output", "indices")`, which are only computed when they are among `$outputs`.
#' For an explanation see [`PipeOpTorch`].
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchMaxPool1D = R6Class("PipeOpTorchMaxPool1D", inherit = PipeOpTorchMaxPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_max_pool1d", param_vals = list()) {
      super$initialize(id = id, d = 1, param_vals = param_vals)
    }
  )
)

#' @title 2D Max Pooling
#' @inherit torch::nnf_max_pool2d description
#' @section nn_module:
#' Calls [`torch::nn_max_pool2d()`] during training.
#' @inheritSection mlr_pipeops_nn_max_pool1d Parameters
#'
#' @templateVar id nn_max_pool2d
#' @inheritSection mlr_pipeops_nn_max_pool1d Input and Output Channels
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchMaxPool2D = R6Class("PipeOpTorchMaxPool2D", inherit = PipeOpTorchMaxPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_max_pool2d", param_vals = list()) {
      super$initialize(id = id, d = 2, param_vals = param_vals)
    }
  )
)


#' @title 3D Max Pooling
#' @inherit torch::nnf_max_pool3d description
#' @section nn_module:
#' Calls [`torch::nn_max_pool3d()`] during training.
#' @inheritSection mlr_pipeops_nn_max_pool1d Parameters
#' @templateVar id nn_max_pool3d
#' @inheritSection mlr_pipeops_nn_max_pool1d Input and Output Channels
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchMaxPool3D = R6Class("PipeOpTorchMaxPool3D", inherit = PipeOpTorchMaxPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_max_pool3d", param_vals = list()) {
      super$initialize(id = id, d = 3, param_vals = param_vals)
    }
  )
)

#' @include aaa.R
register_po("nn_max_pool1d", PipeOpTorchMaxPool1D)
register_po("nn_max_pool2d", PipeOpTorchMaxPool2D)
register_po("nn_max_pool3d", PipeOpTorchMaxPool3D)
