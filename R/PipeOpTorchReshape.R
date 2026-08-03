#' @title Reshape a Tensor
#' @inherit nn_reshape description
#' @section nn_module:
#' Calls [`nn_reshape()`] when trained.
#' This internally calls [`torch::torch_reshape()`] with the given `shape`.
#' @section Parameters:
#' * `shape` :: `integer(1)`\cr
#'   The desired output shape. Unknown dimension (one at most) can either be specified as `-1`.
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
      list(reshape_output_shape(shapes_in[[1L]], param_vals$shape, self$id))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$shape[is.na(param_vals$shape)] = -1
      param_vals
    }
  )
)

#' @title Squeeze a Tensor
#' @inherit nn_squeeze description
#' @section nn_module:
#' Calls [`nn_squeeze()`] when trained.
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   The dimension to squeeze. If `NULL`, all dimensions of size 1 will be squeezed.
#'   Negative values are interpreted downwards from the last dimension.
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
        # An unknown dimension is assumed to not be 1 and is therefore kept:
        # rejecting the shape would rule out networks that are perfectly valid at runtime.
        # `.shape_dependent_params()` passes the squeezed dimensions on to the module, so that
        # the module squeezes exactly those dimensions that are squeezed here.
        return(list(shape[squeeze_dims(shape, keep = TRUE)]))
      }
      # `dim` may select more than one dimension, which `nn_squeeze()` supports
      dim = true_dim
      true_dim = resolve_dim(true_dim, shape)
      # the range is checked before deduplication, so the reported value is the user's
      for (i in seq_along(dim)) {
        assert_dim_in_range(dim[[i]], true_dim[[i]], shape, self$id)
      }
      true_dim = unique(true_dim)

      # an unknown dimension is assumed not to be 1 and is kept, as for `dim = NULL`;
      # `.shape_dependent_params()` pins the module to the dimensions squeezed here
      squeezed = true_dim[!is.na(shape[true_dim]) & shape[true_dim] == 1L]
      if (length(squeezed)) shape = shape[-squeezed]

      list(shape)
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      dim = param_vals[["dim"]]
      if (is.null(dim)) {
        param_vals$dim = squeeze_dims(shapes_in[[1L]])
        return(param_vals)
      }
      # the dimensions that are known to be 1 are selected below, which indexes `shape` and
      # therefore needs positive indices: in R, `shape[-1]` drops the first element instead of
      # addressing the last one
      shape = shapes_in[[1L]]
      dim = resolve_dim(dim, shape)
      # only dimensions that are known to be 1 are squeezed, so that an unknown dimension which
      # happens to be 1 at runtime does not disagree with the inferred shape
      param_vals$dim = unique(dim[!is.na(shape[dim]) & shape[dim] == 1L])
      param_vals
    }
  )
)

# The dimensions that `nn_squeeze` removes when no `dim` is given: those that are known to be 1,
# except the batch dimension (which is never squeezed, because its size is a property of the batch
# and not of the network).
# With `keep = TRUE` the complementary logical vector is returned, i.e. the dimensions that remain.
squeeze_dims = function(shape, keep = FALSE) {
  squeezed = !is.na(shape) & shape == 1
  squeezed[1L] = FALSE
  if (keep) !squeezed else which(squeezed)
}

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
  initialize = function(dim = NULL) {
    self$dim = if (is.null(dim)) NULL else assert_integerish(dim, coerce = TRUE)
  },
  forward = function(input) {
    dims = if (is.null(self$dim)) {
      # All dimensions of size 1 are squeezed, except the batch dimension: `PipeOpTorchSqueeze`
      # cannot know the batch size, so squeezing it would not match the inferred output shape.
      which(input$shape[-1L] == 1L) + 1L
    } else {
      self$dim
    }
    # all dimensions are removed in one call, so the indices are resolved against `input` and not
    # against a tensor that previous removals already shrunk
    input$squeeze(dims)
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
