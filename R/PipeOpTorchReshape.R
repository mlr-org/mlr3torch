#' @title Reshape a Tensor
#' @inherit nn_reshape description
#' @section nn_module:
#' Calls [`nn_reshape()`] when trained.
#' This internally calls [`torch::torch_reshape()`] with the given `shape`.
#' @section Parameters:
#' * `shape` :: `integer()` | `function()`\cr
#'   The desired output shape. One dimension at most can be `-1`, which torch infers from the
#'   number of elements. The first dimension is the batch dimension.
#'
#'   It can also be a `function(shape)` that is called on the input shape and returns the output
#'   shape, e.g. `\(shape) c(shape[1:2], 10)`. This expresses a reshape for inputs whose sizes are
#'   not known in advance, because the function is called again on the shape of the actual tensor
#'   when the network runs. Note that it is called with a shape that can contain `NA`s during shape
#'   inference.
#'   This is e.g. useful when there are multiple unknown dimensions such as `(batch, sequence, ...)`.
#' @templateVar id nn_reshape
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchReshape = R6Class("PipeOpTorchReshape",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_reshape", param_vals = list()) {
      param_set = ps(
        shape = p_uty(tags = c("train", "required"), custom_check = crate(function(x) {
          if (is.function(x)) {
            return(check_function(x, nargs = 1L))
          }
          check_integerish(x, min.len = 1L)
        }))
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
      list(reshape_output_shape(shapes_in[[1L]], param_vals$shape, self$id))
    }
  )
)

#' @title Squeeze a Tensor
#' @inherit nn_squeeze description
#' @section nn_module:
#' Calls [`nn_squeeze()`] when trained.
#' @section Parameters:
#' * `dim` :: `integer()`\cr
#'   The dimensions to squeeze.
#'   Negative values are interpreted downwards from the last dimension.
#'   A dimension whose size is not known to be 1 is kept, because it may be larger at runtime.
#' @templateVar id nn_squeeze
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchSqueeze = R6Class("PipeOpTorchSqueeze",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_squeeze", param_vals = list()) {
      param_set = ps(dim = p_uty(tags = c("train", "required"),
        custom_check = crate(function(x) check_integerish(x, min.len = 1L, any.missing = FALSE))))

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
      dim = param_vals$dim
      true_dim = dim
      true_dim = resolve_dim(true_dim, shape)
      for (i in seq_along(dim)) {
        assert_dim_in_range(dim[[i]], true_dim[[i]], shape, self$id)
      }
      true_dim = unique(true_dim)

      # an unknown dimension is assumed not to be 1 and is kept;
      squeezed = true_dim[!is.na(shape[true_dim]) & shape[true_dim] == 1L]
      if (length(squeezed)) shape = shape[-squeezed]

      list(shape)
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      dim = param_vals[["dim"]]
      shape = shapes_in[[1L]]
      dim = resolve_dim(dim, shape)
      # Only squeeze those that are definitely 1
      param_vals$dim = unique(dim[!is.na(shape[dim]) & shape[dim] == 1L])
      param_vals
    }
  )
)

#' @title Unqueeze a Tensor
#' @inherit nn_unsqueeze description
#' @section nn_module:
#' Calls [`nn_unsqueeze()`] when trained.
#' This internally calls [`torch::torch_unsqueeze()`].
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   The dimension which to unsqueeze. Negative values are interpreted downwards from the last dimension.
#'
#' @templateVar id nn_unsqueeze
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
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
      dim = param_vals[["dim"]]
      # -1 appends a new last dimension, -2 inserts before the last one, etc.
      true_dim = resolve_dim(dim, shape, insert = TRUE)
      # a new dimension may also be appended, hence the shape padded by one
      if (true_dim < 1L || true_dim > length(shape) + 1L) {
        stopf("PipeOp '%s' cannot use 'dim' %i for the input shape %s: a new dimension can be inserted at positions 1 to %i.", # nolint
          self$id, dim, shape_to_str(shape), length(shape) + 1L)
      }
      list(as.integer(append(shape, 1L, after = true_dim - 1)))
    }
  )
)


#' @title Flattens a Tensor
#' @inherit torch::nn_flatten description
#' @section nn_module:
#' Calls [`torch::nn_flatten()`] when trained.
#' @section Parameters:
#' `start_dim` :: `integer(1)`\cr
#'   At wich dimension to start flattening. Default is 2.
#' `end_dim` :: `integer(1)`\cr
#'   At wich dimension to stop flattening. Default is -1.
#'
#' @templateVar id nn_flatten
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchFlatten = R6Class("PipeOpTorchFlatten",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_flatten", param_vals = list()) {
      param_set = ps(
        # negative values count down from the last dimension, `.shapes_out()` checks the range
        start_dim = p_int(default = 2L, tags = "train"),
        end_dim = p_int(default = -1L, tags = "train")
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
      start = param_vals[["start_dim"]] %??% 2L
      end = param_vals[["end_dim"]] %??% -1L

      start_dim = resolve_dim(start, shape)
      end_dim = resolve_dim(end, shape)
      assert_dim_in_range(start, start_dim, shape, self$id)
      assert_dim_in_range(end, end_dim, shape, self$id)
      if (end_dim < start_dim) {
        stopf("PipeOp '%s' requires 'end_dim' (dimension %i) to be at least 'start_dim' (dimension %i).", # nolint
          self$id, end_dim, start_dim)
      }

      list(as.integer(c(shape[seq_len(start_dim - 1)], prod(shape[start_dim:end_dim]), shape[seq_len(length(shape) - end_dim) + end_dim]))) # nolint
    }
  )
)

#' @title Unflattens a Tensor
#' @inherit torch::nn_unflatten description
#' @section nn_module:
#' Calls [`torch::nn_unflatten()`] when trained.
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   The dimension to unflatten. Negative values are interpreted downwards from the last dimension.
#' * `unflattened_size` :: `integer()`\cr
#'   The sizes that replace that dimension. One of them at most can be `-1`, which torch infers
#'   from the size of the dimension being unflattened.
#'
#' @templateVar id nn_unflatten
#' @templateVar param_vals dim = 2, unflattened_size = c(2, 2)
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchUnflatten = R6Class("PipeOpTorchUnflatten",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_unflatten", param_vals = list()) {
      param_set = ps(
        # negative values count down from the last dimension, `.shapes_out()` checks the range
        dim = p_int(tags = c("train", "required")),
        unflattened_size = p_uty(tags = c("train", "required"), custom_check = crate(function(x) {
          check_integerish(x, min.len = 1L, any.missing = FALSE)
        }))
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_unflatten
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      dim = param_vals[["dim"]]
      true_dim = resolve_dim(dim, shape)
      assert_dim_in_range(dim, true_dim, shape, self$id)
      # splitting the batch dimension would silently change the number of observations
      assert_not_batch_dim(true_dim, shape, self$id)
      size = unflatten_output_size(shape[[true_dim]], as.integer(param_vals[["unflattened_size"]]),
        self$id)
      list(as.integer(c(shape[seq_len(true_dim - 1L)], size,
        shape[seq_len(length(shape) - true_dim) + true_dim])))
    }
  )
)

# The sizes that replace the unflattened dimension. `-1` marks the one that torch infers from the
# size of that dimension, which stays unknown as long as that size is.
# @param size_in (`integer(1)`) The size of the dimension being unflattened, possibly `NA`.
# @param target (`integer()`) The `unflattened_size` parameter.
# @param id (`character(1)`) The PipeOp's id, for the error messages.
unflatten_output_size = function(size_in, target, id) {
  # `-1` marks the inferred size, every other entry is a size in its own right
  if (any(target != -1L & target < 1L)) {
    stopf("PipeOp '%s': 'unflattened_size' %s is invalid: every size must be at least 1, only -1 marks the size that is inferred.", # nolint
      id, shape_to_str(target))
  }
  inferred = which(target == -1L)
  # torch can only infer one size, so more than one is invalid whatever the input shape is
  if (length(inferred) > 1L) {
    stopf("PipeOp '%s': 'unflattened_size' %s is invalid: at most one size can be inferred.",
      id, shape_to_str(target))
  }
  out = target
  out[inferred] = NA_integer_
  if (is.na(size_in)) {
    return(out)
  }
  # note that `out[-integer(0)]` is empty, so the inferred size cannot be dropped by negative index
  known = prod(out[setdiff(seq_along(out), inferred)])
  # with an inferred size the known ones must divide the dimension, without one they must multiply
  # out to exactly its size
  ok = if (length(inferred)) size_in %% known == 0 else size_in == known
  if (!ok) {
    stopf("PipeOp '%s': 'unflattened_size' %s does not multiply out to %i, the size of the dimension it unflattens.", # nolint
      id, shape_to_str(target), size_in)
  }
  if (length(inferred)) out[inferred] = size_in / known
  as.integer(out)
}

#' @title Reshape
#'
#' @description Reshape a tensor to the given shape.
#' @param shape (`integer()` | `function()`)\cr
#'   The desired output shape, or a `function(shape)` that is called on the shape of the input
#'   tensor and returns it.
#' @export
nn_reshape = nn_module(
  "nn_reshape",
  initialize = function(shape) {
    if (!is.function(shape)) assert_integerish(shape, min.len = 1L)
    self$shape = shape
  },
  forward = function(input) {
    # a function is called on the shape of the tensor at hand, so that dimensions which are not
    # known when the network is built can be used
    shape = if (is.function(self$shape)) self$shape(dim(input)) else self$shape
    input$reshape(shape)
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
    # `PipeOpTorchSqueeze` passes only the dimensions that are known to be 1, which can be none:
    # squeezing nothing is then the operation that matches the inferred shape
    self$dim = assert_integerish(dim, any.missing = FALSE, coerce = TRUE)
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

#' @include aaa.R
register_po("nn_reshape", PipeOpTorchReshape)
register_po("nn_unsqueeze", PipeOpTorchUnsqueeze)
register_po("nn_squeeze", PipeOpTorchSqueeze)
register_po("nn_flatten", PipeOpTorchFlatten)
register_po("nn_unflatten", PipeOpTorchUnflatten)
