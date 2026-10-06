# Base class for the softmax family: the operators differ only in the module they build, they all
# take the dimension they normalize over and leave the shape alone.
PipeOpTorchSoftmaxDim = R6Class("PipeOpTorchSoftmaxDim",
  inherit = PipeOpTorch,
  public = list(
    # @description
    # Creates a new instance of this [R6][R6::R6Class] class.
    # @template params_pipelines
    # @template param_module_generator
    initialize = function(id, module_generator, param_vals = list()) {
      param_set = ps(
        # negative values count down from the last dimension
        dim = p_int(tags = c("train", "required"))
      )
      super$initialize(
        id = id,
        module_generator = module_generator,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      dim = param_vals[["dim"]]
      assert_dim_in_range(dim, resolve_dim(dim, shape), shape, self$id)
      shapes_in
    }
  )
)

#' @title Softmax
#' @inherit torch::nnf_softmax description
#' @section nn_module:
#' Calls [`torch::nn_softmax()`] when trained.
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
#'
#' @templateVar id nn_softmax
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchSoftmax = R6::R6Class("PipeOpTorchSoftmax",
  inherit = PipeOpTorchSoftmaxDim,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_softmax", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_softmax, param_vals = param_vals)
    }
  )
)

#' @title Softmin
#' @inherit torch::nnf_softmin description
#' @section nn_module:
#' Calls [`torch::nn_softmin()`] when trained.
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   A dimension along which Softmin will be computed (so every slice along dim will sum to 1).
#'
#' @templateVar id nn_softmin
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchSoftmin = R6Class("PipeOpTorchSoftmin",
  inherit = PipeOpTorchSoftmaxDim,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_softmin", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_softmin, param_vals = param_vals)
    }
  )
)

#' @title Log Softmax
#' @inherit torch::nnf_log_softmax description
#' @section nn_module:
#' Calls [`torch::nn_log_softmax()`] when trained.
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   A dimension along which LogSoftmax will be computed.
#'
#' @templateVar id nn_log_softmax
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchLogSoftmax = R6Class("PipeOpTorchLogSoftmax",
  inherit = PipeOpTorchSoftmaxDim,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_log_softmax", param_vals = list()) {
      super$initialize(id = id, module_generator = nn_log_softmax, param_vals = param_vals)
    }
  )
)

#' @title 2D Softmax
#' @inherit torch::nn_softmax2d description
#' @section nn_module:
#' Calls [`torch::nn_softmax2d()`] when trained.
#' The softmax is applied over the feature dimension (dimension 2), so that every spatial position
#' sums to 1.
#' @section Parameters:
#' No parameters.
#'
#' @templateVar id nn_softmax2d
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchSoftmax2D = R6Class("PipeOpTorchSoftmax2D",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_softmax2d", param_vals = list()) {
      super$initialize(
        id = id,
        module_generator = nn_softmax2d,
        param_set = ps(),
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      # torch also accepts a rank-3 input, which it reads as an unbatched (channel, height, width)
      # tensor; here the first dimension is always the batch, so such an input would be normalized
      # over the wrong dimension
      assert_ndim(shapes_in[[1L]], 4L, self$id)
      shapes_in
    }
  )
)

#' @include aaa.R
register_po("nn_softmax", PipeOpTorchSoftmax)
register_po("nn_softmin", PipeOpTorchSoftmin)
register_po("nn_log_softmax", PipeOpTorchLogSoftmax)
register_po("nn_softmax2d", PipeOpTorchSoftmax2D)
