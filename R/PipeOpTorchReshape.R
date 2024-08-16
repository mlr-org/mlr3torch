#' @title Reshape a Tensor
#'
#' @templateVar id nn_reshape
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit nn_reshape description
#'
#' @section Parameters:
#' * `shape` :: `integer(1)`\cr
#'   The desired output shape. Unknown dimension (one at most) can either be specified as `-1` or `NA`.
#'
#' @section Internals:
#' Calls [`nn_reshape()`] when trained.
#' This internally calls [`torch::torch_reshape()`] with the given `shape`.
#' @export
PipeOpTorchReshape = R6Class("PipeOpTorchReshape",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_reshape", param_vals = list()) {
      param_set = ps(
        shape = p_uty(tags = c("train", "required"), custom_check = check_integerish)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_reshape
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = param_vals$shape
      shape[shape == -1] = NA
      inlen = prod(shapes_in[[1]])
      outlen = prod(shape)
      # the following is going to trigger rarely, since the 1st dimension is typically NA
      if (!is.na(inlen) && !is.na(outlen) && inlen != outlen) stop("'shape' not compatible with input shape")
      list(shape)
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$shape[is.na(param_vals$shape)] = -1
      param_vals
    }
  )
)

#' @title Squeeze a Tensor
#'
#' @templateVar id nn_squeeze
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit nn_squeeze description
#'
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   The dimension to squeeze. If `NULL`, all dimensions of size 1 will be squeezed.
#'   Negative values are interpreted downwards from the last dimension.
#' @section Internals:
#' Calls [`nn_squeeze()`] when trained.
#' @export
PipeOpTorchSqueeze = R6Class("PipeOpTorchSqueeze",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_squeeze", param_vals = list()) {
      param_set = ps(dim = p_uty(tags = "train", custom_check = check_integerish_or_null))

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_squeeze
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1]]
      true_dim = param_vals$dim

      if (is.null(true_dim)) {
        # if dim is left unspecified we squeeze everything.
        shape = shape[shape != 1]
        if (length(shape) < 2) {
          stopf("Output tensor would have less than (<) 2 dimensions.")
        }
        return(list(shape))
      } else if (true_dim < 0) { # start counting downwards from the last dimension
        true_dim = 1 + length(shape) + true_dim
      }
      assert_int(true_dim, lower = 1, upper = length(shape))

      if (is.na(shape[[true_dim]])) stop("input shape for 'dim' dimension must be known.")
      if (shape[[true_dim]] == 1) shape = shape[-true_dim]

      list(shape)
    }
  )
)

#' @title Unqueeze a Tensor
#'
#' @templateVar id nn_unsqueeze
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   The dimension which to unsqueeze. Negative values are interpreted downwards from the last dimension.
#'
#' @section Internals:
#' Calls [`nn_unsqueeze()`] when trained.
#' This internally calls [`torch::torch_unsqueeze()`].
#' @export
PipeOpTorchUnsqueeze = R6Class("PipeOpTorchUnsqueeze",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_unsqueeze", param_vals = list()) {
      param_set = ps(dim = p_int(tags = c("train", "required")))
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_unsqueeze
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1]]
      true_dim = param_vals$dim
      if (true_dim < 0) {
        true_dim = 1 + length(shape) - true_dim
      }
      assert_int(true_dim, lower = 1, upper = length(shape) + 1)
      list(append(shape, 1, after = true_dim - 1))
    }
  )
)


#' @title Flattens a Tensor
#'
#' @templateVar id nn_flatten
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit torch::nn_flatten description
#'
#' @section Parameters:
#' `start_dim` :: `integer(1)`\cr
#'   At wich dimension to start flattening. Default is 2.
#' `end_dim` :: `integer(1)`\cr
#'   At wich dimension to stop flattening. Default is -1.
#' @section Internals:
#' Calls [`torch::nn_flatten()`] when trained.
#' @export
PipeOpTorchFlatten = R6Class("PipeOpTorchFlatten",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_flatten", param_vals = list()) {
      param_set = ps(
        start_dim = p_int(default = 2L, lower = 1L, tags = "train"),
        end_dim = p_int(default = -1L, lower = 1L, tags = "train", special_vals = list(-1L))
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_flatten
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1]]
      start_dim = param_vals$start_dim %??% 2
      end_dim = param_vals$end_dim %??% -1

      if (start_dim < 0) start_dim = 1 + length(shape) + start_dim
      if (end_dim < 0) end_dim = 1 + length(shape) + end_dim
      assert_int(start_dim, lower = 1, upper = length(shape))
      assert_int(end_dim, lower = start_dim, upper = length(shape))

      list(c(shape[seq_len(start_dim - 1)], prod(shape[start_dim:end_dim]), shape[seq_len(length(shape) - end_dim) + end_dim])) # nolint
    }
  )
)

#' @title Reshape
#'
#' @description Reshape a tensor to the given shape.
#' @param shape (`integer()`)\cr
#'   The desired output shape.
#' @export
nn_reshape = nn_module(
  "nn_reshape",
  initialize = function(shape) {
    assert_integerish(shape)
    self$shape = shape
  },
  forward = function(input) {
    input$reshape(self$shape)
  }
)

#' @title Squeeze
#'
#' @description Squeezes a tensor by calling [`torch::torch_squeeze()`] with the given dimension `dim`.
#' @param dim (`integer()`)\cr
#'   The dimension to squeeze.
#' @export
nn_squeeze = nn_module(
  "nn_squeeze",
  initialize = function(dim) {
    assert_int(dim)
    self$dim = dim
  },
  forward = function(input) {
    input$squeeze(self$dim)
  }
)

#' @title Unsqueeze
#'
#' @description Unsqueezes a tensor by calling [`torch::torch_unsqueeze()`] with the given dimension `dim`.
#' @param dim (`integer(1)`)\cr
#'   The dimension to unsqueeze.
#' @export
nn_unsqueeze = nn_module(
  "nn_unsqueeze",
  initialize = function(dim) {
    assert_int(dim)
    self$dim = dim
  },
  forward = function(input) {
    input$unsqueeze(self$dim)
  }
)

#' @include zzz.R
register_po("nn_reshape", PipeOpTorchReshape)
register_po("nn_unsqueeze", PipeOpTorchUnsqueeze)
register_po("nn_squeeze", PipeOpTorchSqueeze)
register_po("nn_flatten", PipeOpTorchFlatten)
