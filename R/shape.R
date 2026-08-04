#' @title Assert a Shape
#'
#' @description
#' Checks that `shape` is a valid shape, i.e. an `integer()` with at least one dimension, `NA` where
#' a dimension is unknown. See the "Shape Inference" section of [`PipeOpTorch`] for the conventions.
#'
#' @param shape (`integer()`)\cr
#'   A shape, with `NA` for the dimensions whose size is unknown.
#' @param null_ok (`logical(1)`)\cr
#'   Whether `NULL`, i.e. a wholly unknown shape, is valid.
#' @param coerce (`logical(1)`)\cr
#'   Whether to coerce the input to an `integer()` if possible.
#' @param unknown_batch (`logical(1)` | `NULL`)\cr
#'   Whether the batch dimension **must** be unknown, i.e. `NA`.
#'   If left `NULL` (default), the first dimension can be `NA` or not.
#' @param len (`integer(1)`)\cr
#'   The required number of dimensions.
#'
#' @return The shape, coerced to an `integer()` if `coerce` is `TRUE`.
#'
#' @family Shape Assertions
#' @examples
#' assert_shape(c(NA, 3, 32, 32))
#' try(assert_shape("not a shape"))
#' @export
assert_shape = function(shape, null_ok = FALSE, coerce = TRUE, unknown_batch = NULL, len = NULL) {
  result = check_shape(shape, null_ok = null_ok, unknown_batch = unknown_batch, len = len)

  if (!isTRUE(result)) stopf(result)

  if (coerce && !is.null(shape)) {
    return(as.integer(shape))
  }
  shape
}


test_shape = function(shape, null_ok = FALSE, unknown_batch = NULL, len = NULL) {
  if (is.null(shape) && null_ok) {
    return(TRUE)
  }
  ok = test_integerish(shape, min.len = 1L, any.missing = TRUE, len = len)

  if (!ok) {
    return(FALSE)
  }

  if (is.null(unknown_batch)) {
    # first dim can be present or missing
    return(TRUE)
  }
  return(is.na(shape[1L]) == unknown_batch)
}

check_shape = function(x, null_ok = FALSE, unknown_batch = NULL, len = NULL) {
  if (test_shape(x, null_ok = null_ok, unknown_batch = unknown_batch, len = len)) {
    return(TRUE)
  }
  # `shape_to_str()` only formats shapes and `list()`s of them, so anything else -- a string, say --
  # would fail its own assertion and hide the message we actually want to give here
  if (!is.numeric(x) && !is.logical(x) && !is.list(x) && !is.null(x)) {
    return(sprintf("Invalid shape: must be an integer vector, but is %s.", class(x)[1L]))
  }
  sprintf("Invalid shape: %s.", shape_to_str(x))
}

#' @title Assert a List of Shapes
#'
#' @description
#' Checks that `shapes` is a non-empty `list()` of valid shapes, see [`assert_shape()`].
#'
#' @param shapes (`list()` of `integer()`)\cr
#'   A `list()` of shapes.
#' @param coerce (`logical(1)`)\cr
#'   Whether to coerce the shapes to `integer()` if possible.
#' @param named (`logical(1)`)\cr
#'   Whether the shapes must be uniquely named.
#' @param null_ok (`logical(1)`)\cr
#'   Whether `NULL`, i.e. a wholly unknown shape, is valid.
#' @param unknown_batch (`logical(1)` | `NULL`)\cr
#'   Whether the batch dimension **must** be unknown, i.e. `NA`.
#'   If left `NULL` (default), the first dimension can be `NA` or not.
#'
#' @return The shapes, coerced to `integer()` if `coerce` is `TRUE`.
#'
#' @family Shape Assertions
#' @examples
#' assert_shapes(list(c(NA, 3), c(NA, 5)))
#' @export
assert_shapes = function(shapes, coerce = TRUE, named = FALSE, null_ok = FALSE, unknown_batch = NULL) {
  ok = test_list(shapes, min.len = 1L)
  if (named) {
    assert_names(setdiff(names(shapes), "..."), type = "unique")
  }
  if (!ok) {
    stopf("Invalid shape")
  }
  map(shapes, assert_shape, coerce = coerce, null_ok = null_ok, unknown_batch = unknown_batch)
}

#' @title Assert that Dimensions are Known
#'
#' @description
#' Ensure that a specific dimension is known, i.e. not `NA`.
#'
#' @param shape (`integer()`)\cr
#'   The input shape.
#' @param dims (`integer()`)\cr
#'   Indices of the dimensions that must be known.
#' @param what (`character(1)`)\cr
#'   Describes those dimensions in the error message, e.g. `"the channel dimension (dimension 2)"`.
#' @param id (`character(1)` | `NULL`)\cr
#'   The id of the [`PipeOp`][mlr3pipelines::PipeOp] the assertion is made for, which the error
#'   message names. `NULL` when the assertion is not made for a `PipeOp`.
#'
#' @return The shape, invisibly.
#'
#' @family Shape Assertions
#' @examples
#' # a convolution needs the number of input channels, but not the spatial extent
#' assert_known_dims(c(NA, 3, NA, NA), 2, "the number of channels", id = "nn_conv2d")
#' try(assert_known_dims(c(NA, NA, 10, 10), 2, "the number of channels", id = "nn_conv2d"))
#' @export
assert_known_dims = function(shape, dims, what, id = NULL) {
  if (!anyNA(shape[dims])) {
    return(invisible(shape))
  }
  if (is.null(id)) {
    stopf("Expected %s of the input shape to be known, but got shape %s.", what, shape_to_str(shape))
  }
  stopf("PipeOp '%s' requires %s of the input shape to be known, but got shape %s.",
    id, what, shape_to_str(shape))
}

#' @title Helpers for Shape Inference
#'
#' @name shape_helpers
#'
#' @description
#' Helpers for writing the `private$.shapes_out()` method of a [`PipeOpTorch`]. They all propagate
#' unknown (`NA`) dimensions, see the "Shape Inference" section of [`PipeOpTorch`] for the shape conventions.
#'
#' @details
#' * `broadcast_shapes()` applies the broadcasting rules of `torch`, generalized to shapes that may
#'   contain `NA`. Per dimension a known size that is not 1 wins; if all known sizes are 1 and some
#'   input is unknown, the result is unknown, because the unknown one may be greater than 1 and
#'   would then determine the size. The shapes must already have the same number of dimensions:
#'   shorter ones are not left-padded with 1s, because the first dimension is the batch dimension.
#' * `resolve_dim()` resolves dimension indices that count from the end, as in `torch`, to positive
#'   ones. Indices that are out of range stay out of range, so that
#'   [`assert_dim_in_range()`] reports them.
#' * `shape_to_str()` formats a shape, or a `list()` of them, for an error message.
#'
#' @param shapes (`list()` of `integer()`)\cr
#'   The input shapes, all with the same number of dimensions.
#' @param shape (`integer()`)\cr
#'   The input shape.
#' @param dim (`integer()`)\cr
#'   The dimension(s) the operator addresses. Negative ones count back from the last dimension:
#'   `-1` is the last dimension, `-2` the one before it, and so on.
#' @param insert (`logical(1)`)\cr
#'   Whether a dimension is inserted rather than addressed, as by
#'   [`nn_unsqueeze()`][mlr_pipeops_nn_unsqueeze]: there is then one more position than the shape
#'   has dimensions, `-1` appending a new last one.
#' @param id (`character(1)`)\cr
#'   The id of the [`PipeOp`][mlr3pipelines::PipeOp], which the error message names.
#' @param x (`integer()` | `list()` of `integer()` | `NULL`)\cr
#'   The shape(s) to format. `NULL` stands for an unknown shape.
#'
#' @return
#' `broadcast_shapes()` returns an `integer()` shape, `resolve_dim()` an `integer()` of the same
#' length as `dim`, and `shape_to_str()` a `character(1)`.
#'
#' @family Shape Inference
#' @examples
#' broadcast_shapes(list(c(NA, 1), c(NA, 6)), id = "nn_merge_sum")
#' resolve_dim(-1, c(NA, 3, 8))
#' shape_to_str(c(NA, 3, 8))
NULL

# Dimension-wise result of concatenating `shapes`, where the concatenated dimension is left as `NA`
# for the caller to fill in.
# Unlike broadcasting, `torch_cat()` requires all other dimensions to be equal, so a known size
# of 1 does not combine with a different known size. Rejecting that here fails when the network is
# built instead of when it is run.
# @param shapes (`list()` of `integer()`) The input shapes, all with the same number of dimensions.
# @param dim (`integer(1)`) The concatenated dimension, resolved to a positive index.
# @param id (`character(1)`) The PipeOp's id, for the error message.
cat_shapes = function(shapes, dim, id) {
  mat = do.call(rbind, shapes)
  as.integer(map_int(seq_len(ncol(mat)), function(i) {
    if (i == dim) {
      return(NA_integer_)
    }
    known = unique(mat[, i][!is.na(mat[, i])])
    if (length(known) > 1L) {
      stopf("PipeOp '%s' cannot concatenate its input shapes %s: dimension %i has the sizes %s. All dimensions except the concatenated dimension %i must be equal.", # nolint
        id, shape_to_str(shapes), i, paste0(known, collapse = " and "), dim)
    }
    if (length(known)) as.integer(known) else NA_integer_
  }))
}

#' @rdname shape_helpers
#' @export
broadcast_shapes = function(shapes, id) {
  mat = do.call(rbind, shapes)
  as.integer(map_int(seq_len(ncol(mat)), function(i) {
    vals = mat[, i]
    known = unique(vals[!is.na(vals)])
    non1 = known[known != 1L]
    if (length(non1) > 1L) {
      stopf("PipeOp '%s' cannot broadcast its input shapes %s: dimension %i has the incompatible sizes %s. Dimensions must be equal or 1.", # nolint
        id, shape_to_str(shapes), i, paste0(non1, collapse = " and "))
    }
    if (length(non1) == 1L) {
      return(as.integer(non1))
    }
    if (anyNA(vals)) NA_integer_ else 1L
  }))
}

# The target shape of a reshape, which may be given as a function of the input shape so that it can
# be expressed for inputs whose sizes are not known in advance, e.g. `\(shape) c(shape[1:2], 10)`.
# The function is called both here, on the shape the inference knows (which may contain `NA`), and
# in `nn_reshape()`, on the shape of the tensor at hand.
# @param shape (`integer()` | `function()`) The target shape or a function returning one.
# @param shape_in (`integer()`) The input shape the function is called on.
# @param id (`character(1)`) The PipeOp's id, for the error messages.
resolve_shape_param = function(shape, shape_in, id) {
  if (is.function(shape)) {
    target = shape(shape_in)
    if (!test_integerish(target, min.len = 1L)) {
      stopf("PipeOp '%s': 'shape' returned '%s' for the input shape %s, but must return at least one dimension, each of which is a number or `NA`.", # nolint
        id, paste0(format(target), collapse = ","), shape_to_str(shape_in))
    }
    return(as.integer(target))
  }
  if (!length(shape)) {
    stopf("PipeOp '%s' requires 'shape' to have at least one dimension.", id)
  }
  if (anyNA(shape)) {
    stopf("PipeOp '%s': 'shape' %s is invalid: use -1 for the dimension that is inferred from the number of elements.", # nolint
      id, shape_to_str(shape))
  }
  as.integer(shape)
}

#' @title Output Shape of a Reshape
#'
#' @description
#' The shape of [`torch_reshape(x, shape)`][torch::torch_reshape], which is what
#' [`nn_reshape()`][mlr_pipeops_nn_reshape] infers. The dimension that `torch` infers from the
#' number of elements is resolved here whenever that number is known, and stays unknown otherwise.
#'
#' @param shape_in (`integer()`)\cr
#'   The input shape, with `NA` for the dimensions whose size is unknown.
#' @param shape (`integer()` | `function()`)\cr
#'   The target shape, where `-1` marks the dimension that `torch` infers from the number of
#'   elements. A `function(shape)` of the input shape is called on it and must return such a vector,
#'   which lets a reshape be expressed for inputs whose sizes are not known in advance, e.g.
#'   `\(shape) c(shape[1:2], 10)`.
#' @param id (`character(1)`)\cr
#'   The id of the [`PipeOp`][mlr3pipelines::PipeOp], which the error messages name.
#'
#' @return (`integer()`) The output shape.
#'
#' @family Shape Inference
#' @examples
#' reshape_output_shape(c(NA, 3, 4), c(-1, 12), id = "nn_reshape")
#' reshape_output_shape(c(NA, 3, 4), \(shape) c(shape[1], 12), id = "nn_reshape")
#' @export
reshape_output_shape = function(shape_in, shape, id) {
  # resolve target if shape is a function
  target = resolve_shape_param(shape, shape_in, id)
  assert_reshape_target(target, id)

  # The number of input elements is unknown as soon as one dimension is (typically the batch), and
  # so is the number of output elements when the target itself contains an unknown size. The
  # inferred dimension then stays unknown, and with an unknown element count on either side there
  # is nothing left that could be checked.
  if (is.na(prod(shape_in)) || anyNA(target)) {
    # replaces -1 with NA
    return(as.integer(reshape_fill_inferred_per_observation(shape_in, target)))
  }

  # Here we have a concrete shape
  out = reshape_fill_inferred_known(shape_in, target, id)
  assert_reshape_keeps_batch(shape_in, out, target, id)
  as.integer(out)
}

# The total number of elements is unknown as soon as any dimension is, but the number of elements
# *per observation* may still be known -- it is whenever the batch dimension is the only unknown one.
# That is enough to resolve an inferred dimension sitting after the batch dimension, which is what a
# `function(shape)` target that keeps the batch and flattens the rest produces, e.g.
# `\(shape) c(shape[1], -1)`. Without this the dimension would stay unknown and the next operator,
# which usually needs to know its number of input features, could not be built at all.
# @param shape_in (`integer()`) The input shape.
# @param target (`numeric()`) The target shape, see `resolve_shape_param()`.
reshape_fill_inferred_per_observation = function(shape_in, target) {
  out = reshape_target_as_shape(target)
  inferred = which(!is.na(target) & target == -1)
  # `-1` in the first position is the batch dimension itself, which no number of elements per
  # observation can pin down; a known first entry means the target does not keep the batch anyway
  if (length(inferred) != 1L || inferred == 1L || !is.na(target[[1L]])) {
    return(out)
  }
  per_obs = prod(shape_in[-1L])
  known = out[-c(1L, inferred)]
  if (is.na(per_obs) || anyNA(known)) {
    return(out)
  }
  # a target that does not divide the input cannot be right for any batch size, but reporting the
  # dimension as unknown is the permissive answer and leaves the complaint to the runtime
  rest = prod(known)
  if (rest > 0 && per_obs %% rest == 0) {
    out[inferred] = per_obs / rest
  }
  out
}

reshape_target_as_shape = function(target) {
  target[!is.na(target) & target == -1] = NA_integer_
  as.integer(target)
}

# Rejects a target that no tensor can have, whatever the input shape turns out to be.
# @param target (`numeric()`) The target shape, see `resolve_shape_param()`.
# @param id (`character(1)`) The PipeOp's id, for the error messages.
assert_reshape_target = function(target, id) {
  # `-1` marks the inferred dimension, every other entry is a size in its own right
  if (any(!is.na(target) & target != -1 & target < 1)) {
    stopf("PipeOp '%s': 'shape' %s is invalid: every dimension must be at least 1, only -1 marks the dimension that is inferred from the number of elements.", # nolint
      id, shape_to_str(target))
  }
  # torch can only infer one dimension, so more than one is invalid whatever the input shape is
  if (sum(!is.na(target) & target == -1) > 1L) {
    stopf("PipeOp '%s': 'shape' %s is invalid: at most one dimension can be inferred from the number of elements.", # nolint
      id, shape_to_str(target))
  }
  as.integer(target)
}

# Resolves the inferred dimension from the number of elements and rejects a target whose element
# count cannot match. Both arguments must be fully known -- an `NA` anywhere makes the element
# counts below `NA` and the comparisons undecidable -- which the caller ensures by returning early.
# @param shape_in (`integer()`) The input shape, without unknown dimensions.
# @param target (`numeric()`) The target shape, without unknown dimensions, see
#   `resolve_shape_param()`. `-1` marks the dimension that is resolved here.
# @param id (`character(1)`) The PipeOp's id, for the error message.
reshape_fill_inferred_known = function(shape_in, target, id) {
  out = reshape_target_as_shape(target)
  inferred = which(!is.na(target) & target == -1)
  inlen = prod(shape_in)
  # note that `out[-integer(0)]` is empty, so the two cases have to be distinguished
  knownlen = if (length(inferred)) prod(out[-inferred]) else prod(out)
  # with an inferred dimension the known ones must divide the input, without one the target must
  # have exactly as many elements as the input
  if (knownlen == 0 || (length(inferred) && inlen %% knownlen != 0) ||
      (!length(inferred) && inlen != knownlen)) {
    stopf("PipeOp '%s': 'shape' %s is not compatible with the input shape %s.",
      id, shape_to_str(target), shape_to_str(shape_in))
  }
  if (length(inferred)) {
    out[inferred] = inlen / knownlen
  }
  as.integer(out)
}

# The first dimension is the batch dimension. A reshape that moves elements across it changes the
# batch size, which will cause weird errors (e.g. when computing the loss later).
# @param shape_in (`integer()`) The input shape.
# @param out (`numeric()`) The output shape, with the inferred dimension already filled in.
# @param target (`numeric()`) The target shape, for the error message and to tell whether the batch
#   dimension was meant to be kept at all.
# @param id (`character(1)`) The PipeOp's id, for the error message.
assert_reshape_keeps_batch = function(shape_in, out, target, id) {
  rest_in = prod(shape_in[-1L])
  rest_out = prod(out[-1L])
  # a target that keeps the batch dimension either infers it or repeats the input's; `-1` in the
  # first position is only a batch dimension if nothing else was inferred from it
  keeps_batch = is.na(target[[1L]]) || target[[1L]] == -1 || isTRUE(target[[1L]] == shape_in[[1L]])
  if (keeps_batch && !is.na(rest_in) && !is.na(rest_out) && rest_in != rest_out) {
    stopf("PipeOp '%s': 'shape' %s changes the batch dimension of the input shape %s: it maps %s elements per observation to %s.", # nolint
      id, shape_to_str(target), shape_to_str(shape_in), rest_in, rest_out)
  }
  invisible(out)
}

#' @rdname shape_helpers
#' @export
resolve_dim = function(dim, shape, insert = FALSE) {
  negative = dim < 0
  dim[negative] = length(shape) + as.integer(insert) + 1L + dim[negative]
  dim
}

#' @title Assert that a Dimension Exists
#'
#' @description
#' Rejects a `dim` parameter that does not address a dimension of `shape`. Resolve negative indices
#' with [`resolve_dim()`] first and pass both, so that the error message can report the value the
#' user actually specified.
#'
#' @param dim (`integer(1)`)\cr
#'   The dimension as the user specified it: it may count back from the last dimension. Only used
#'   for the error message.
#' @param true_dim (`integer(1)`)\cr
#'   The same dimension resolved to a positive index, see [`resolve_dim()`].
#' @param shape (`integer()`)\cr
#'   The shape that `true_dim` addresses.
#' @param id (`character(1)`)\cr
#'   The id of the [`PipeOp`][mlr3pipelines::PipeOp], which the error message names.
#'
#' @return The resolved dimension, invisibly.
#'
#' @family Shape Assertions
#' @examples
#' shape = c(NA, 3, 8, 8)
#' assert_dim_in_range(-1L, resolve_dim(-1L, shape), shape, id = "nn_squeeze")
#' try(assert_dim_in_range(-5L, resolve_dim(-5L, shape), shape, id = "nn_squeeze"))
#' @export
assert_dim_in_range = function(dim, true_dim, shape, id) {
  if (true_dim >= 1L && true_dim <= length(shape)) {
    return(invisible(true_dim))
  }
  stopf("PipeOp '%s' cannot use 'dim' %i for the input shape %s, which has %i dimension(s).",
    id, dim, shape_to_str(shape), length(shape))
}

#' @title Assert that a Dimension is not the Batch Dimension
#'
#' @description
#' Rejects an operation on the first dimension, which is the batch dimension. An operator that
#' changes it would silently change the number of observations, which fails much later with a
#' mismatch against the target.
#'
#' @param dim (`integer(1)`)\cr
#'   The resolved dimension that the operator changes, see [`resolve_dim()`].
#' @param shape (`integer()`)\cr
#'   The input shape, used for the error message.
#' @param id (`character(1)`)\cr
#'   The id of the [`PipeOp`][mlr3pipelines::PipeOp], which the error message names.
#'
#' @return The dimension, invisibly.
#'
#' @family Shape Assertions
#' @examples
#' assert_not_batch_dim(2L, c(NA, 3, 8), id = "nn_squeeze")
#' try(assert_not_batch_dim(1L, c(NA, 3, 8), id = "nn_squeeze"))
#' @export
assert_not_batch_dim = function(dim, shape, id) {
  if (dim != 1L) {
    return(invisible(dim))
  }
  stopf("PipeOp '%s' would change dimension 1 of the input shape %s, which is the batch dimension.",
    id, shape_to_str(shape))
}

halve_dim = function(shape, dim, id) {
  halved = shape[dim] / 2
  if (!test_integerish(halved)) {
    stopf("PipeOp '%s' requires dimension %i of the input shape %s to be divisible by 2, but it is %i.", # nolint
      id, dim, shape_to_str(shape), shape[dim])
  }
  shape[dim] = halved
  as.integer(shape)
}

#' @title Assert the Number of Dimensions of a Shape
#'
#' @description
#' Rejects a shape with the wrong number of dimensions. Give either `ndim`, the number(s) of
#' dimensions the operator accepts, or the bounds `min` and `max` (either on its own), which is what
#' an operator that accepts a range of them uses, as batch normalization does.
#'
#' @param shape (`integer()`)\cr
#'   The input shape, with `NA` for the dimensions whose size is unknown.
#' @param ndim (`integer()`)\cr
#'   The number(s) of dimensions the operator accepts, the batch dimension included.
#'   Alternatively give `min` and/or `max`.
#' @param id (`character(1)`)\cr
#'   The id of the [`PipeOp`][mlr3pipelines::PipeOp], which the error message names.
#' @param min (`integer(1)`)\cr
#'   The smallest number of dimensions the operator accepts, `NULL` for no lower bound.
#' @param max (`integer(1)`)\cr
#'   The largest number of dimensions the operator accepts, `NULL` for no upper bound.
#'
#' @return The shape, invisibly.
#'
#' @family Shape Assertions
#' @examples
#' assert_ndim(c(NA, 3, 32, 32), 4L, id = "nn_conv2d")
#' try(assert_ndim(c(NA, 3), 4L, id = "nn_conv2d"))
#' # batch normalization accepts a range
#' try(assert_ndim(c(NA, 3, 4, 5), id = "nn_batch_norm1d", min = 2L, max = 3L))
#' @export
assert_ndim = function(shape, ndim = NULL, id, min = NULL, max = NULL) {
  n = length(shape)
  ok = (is.null(ndim) || n %in% ndim) && (is.null(min) || n >= min) && (is.null(max) || n <= max)
  if (ok) {
    return(invisible(shape))
  }
  # `ndim` and a bound are not combined by any caller, so the message describes whichever was given
  expected = if (!is.null(ndim)) {
    paste0(ndim, collapse = " or ")
  } else if (is.null(max)) {
    sprintf("at least %i", min)
  } else if (is.null(min)) {
    sprintf("at most %i", max)
  } else if (min == max) {
    as.character(min)
  } else if (max - min == 1L) {
    sprintf("%i or %i", min, max)
  } else {
    sprintf("%i to %i", min, max)
  }
  stopf("PipeOp '%s' requires an input with %s dimensions (the first one being the batch dimension), but got the shape %s, which has %i.", # nolint
    id, expected, shape_to_str(shape), n)
}

#' @title Assert that a Computed Output Size is Positive
#'
#' @description
#' Rejects a computed output size that no tensor can have, i.e. one that is not a positive number.
#' Operators such as convolutions and pooling compute their spatial output sizes from `kernel_size`,
#' `stride`, `padding` and `dilation`, a combination of which can produce a size of zero or less.
#' Unknown (`NA`) sizes are accepted, since nothing can be said about them.
#'
#' @param extent (`numeric()`)\cr
#'   The sizes an operator computed for the dimensions it changes, e.g. the spatial dimensions of a
#'   convolution.
#' @param shape_in (`integer()`)\cr
#'   The input shape, used for the error message.
#' @param id (`character(1)` | `NULL`)\cr
#'   The id of the [`PipeOp`][mlr3pipelines::PipeOp], which the error message names.
#'   `NULL` when the assertion is not made for a `PipeOp`.
#'
#' @return The extent, invisibly.
#'
#' @family Shape Assertions
#' @examples
#' assert_positive_extent(c(30, NA), c(NA, 3, 32, 32), id = "nn_conv2d")
#' try(assert_positive_extent(c(0, 4), c(NA, 3, 32, 32), id = "nn_conv2d"))
#' @export
assert_positive_extent = function(extent, shape_in, id) {
  invalid = !is.na(extent) & (!is.finite(extent) | extent < 1)
  if (!any(invalid)) {
    return(invisible(extent))
  }
  stopf("%s cannot be applied to the input shape %s: it would produce an output of size %s, which no tensor can have. Check 'kernel_size', 'stride', 'padding' and 'dilation'.", # nolint
    if (is.null(id)) "The operator" else sprintf("PipeOp '%s'", id), shape_to_str(shape_in),
    paste0(extent, collapse = ", "))
}

#' @title Assert that Shapes have the Same Number of Dimensions
#'
#' @description
#' Rejects inputs that do not all have the same number of dimensions, which the operators that
#' combine several inputs require: shorter shapes are not left-padded with 1s, because the first
#' dimension is the batch dimension.
#'
#' @param shapes (`list()` of `integer()`)\cr
#'   The input shapes.
#' @param id (`character(1)`)\cr
#'   The id of the [`PipeOp`][mlr3pipelines::PipeOp], which the error message names.
#'
#' @return The shapes, invisibly.
#'
#' @family Shape Assertions
#' @examples
#' assert_same_ndim(list(c(NA, 3), c(NA, 5)), id = "nn_merge_cat")
#' try(assert_same_ndim(list(c(NA, 3), c(NA, 5, 2)), id = "nn_merge_cat"))
#' @export
assert_same_ndim = function(shapes, id) {
  ndim = lengths(shapes)
  if (length(unique(ndim)) == 1L) {
    return(invisible(shapes))
  }
  stopf("PipeOp '%s' requires all its inputs to have the same number of dimensions, but got the shapes %s (with %s dimensions).", # nolint
    id, shape_to_str(shapes), paste0(ndim, collapse = ", "))
}

#' @title Assert that Shapes have the Same Batch Size
#'
#' @description
#' Rejects inputs whose batch sizes, i.e. first dimensions, disagree. An unknown (`NA`) batch size
#' is compatible with any other, so only the known ones have to agree.
#'
#' Unlike the other assertions this returns the common batch size rather than its input, because
#' that is what the caller needs next: an operator that drops the batch dimension while it works
#' has to put it back afterwards.
#'
#' @param shapes (`list()` of `integer()`)\cr
#'   The input shapes.
#' @param id (`character(1)`)\cr
#'   The id of the [`PipeOp`][mlr3pipelines::PipeOp], which the error message names.
#'
#' @return (`integer(1)`) The common batch size, invisibly, or `NA_integer_` if no input has a
#'   known one.
#'
#' @family Shape Assertions
#' @examples
#' assert_same_batch_size(list(c(8, 3), c(8, 5)), id = "nn_block")
#' # an unknown batch size is compatible with a known one, which is the one that is returned
#' assert_same_batch_size(list(c(NA, 3), c(8, 5)), id = "nn_block")
#' try(assert_same_batch_size(list(c(8, 3), c(4, 5)), id = "nn_block"))
#' @export
assert_same_batch_size = function(shapes, id) {
  batch = map_int(shapes, function(shape) as.integer(shape[[1L]]))
  known = unique(batch[!is.na(batch)])
  if (length(known) > 1L) {
    stopf("PipeOp '%s' requires all its inputs to have the same batch size, but got the shapes %s (with the batch sizes %s).", # nolint
      id, shape_to_str(shapes), paste0(known, collapse = " and "))
  }
  invisible(if (length(known)) known else NA_integer_)
}




# Sizes that `infer_shapes()` substitutes for *all* `NA`s of a shape, one per trace: a dimension
# that comes out the same for every filling is known, one that varies with it is not.
#
# One small and two large values, because the two ends catch different operators: those with a
# minimum extent (a convolution with a large kernel) fail on a small input, while those that clamp
# to the input size (`x[, 1:32]`, `transform_crop(height = 16)`) look size-preserving unless traced
# small. A filling that fails is dropped by the caller, so covering both ends is free.
# No filling may be 1, which broadcasts and is squeezed away, changing the rank of the trace.
#
# @param shape (`integer()`) The shape whose `NA`s are replaced; only the number of unknown
#   dimensions and the product of the known ones matter.
# @param max_elements (`numeric(1)`) How many elements a traced tensor may have. Smaller fillings
#   are returned when the largest ones would exceed it, i.e. for many unknown or large known
#   dimensions.
# @return (`integer(3)`) The fillings, smallest first.
na_replacements = function(shape, max_elements = 1e7) {
  n_unknown = sum(is.na(shape))
  candidates = list(c(2L, 83L, 89L), c(2L, 23L, 29L), c(2L, 7L, 11L), c(2L, 3L, 5L))
  if (!n_unknown) {
    return(candidates[[1L]])
  }
  n_known = prod(as.numeric(shape[!is.na(shape)]))
  for (candidate in candidates) {
    if (n_known * max(candidate)^n_unknown <= max_elements) {
      return(candidate)
    }
  }
  # even the smallest candidate is above `max_elements`, which happens when the known dimensions
  # are already large; there is nothing smaller to fall back to
  last(candidates)
}

#' @title Infer Shapes
#' @description
#' Infer the shapes of the output of a function based on the shapes of the input.
#' This works by running the function on the input and observing the results.
#' For fully known input shapes this is always correct.
#' For partially unknown shapes, the `NA`s are replaced with various concrete values
#' and the output shape is computed from them.
#' Note that this is a heuristic that might fail, so usually one wants to provide the shape
#' (inference) explicitly.
#'
#' @details
#'
#' The inference is done as follows:
#' 1. All `NA`s are replaced with three different values, which span a wide range: none of them is
#'    `1` (which broadcasts and is squeezed away), one of them is small (to detect operators that
#'    clamp to the input size, such as slicing or cropping) and the others are large (because
#'    operators such as a convolution with a large kernel need a minimum extent).
#' 2. Three tensors are generated for the three shapes of step 1.
#' 3. The function is called on these three tensors and the shapes are calculated.
#'    A call that fails is dropped, so that an operator is not rejected because of the smallest
#'    value; at least two of the three calls must succeed.
#' 4. If:
#'    * the number of dimensions varies, an error is thrown.
#'    * the number of dimensions is the same, values are set to `NA` if the dimension is varying
#'      between the tensors and otherwise set to the unique value.
#'
#' @param shapes_in (`list()`)\cr
#'   A list of shapes of the input tensors.
#' @param param_vals (`list()`)\cr
#'   A list of named parameters for the function.
#' @param output_names (`character()`)\cr
#'   The names of the output tensors.
#' @param fn (`function()`)\cr
#'   The function to infer the shapes for.
#' @param rowwise (`logical(1)`)\cr
#'   Whether the function is rowwise.
#' @param id (`character(1)`)\cr
#'   The id of the PipeOp (for error messages).
#' @return (`list()`)\cr
#'   A list of shapes of the output tensors.
#' @family Shape Inference
#' @export
infer_shapes = function(shapes_in, param_vals, output_names, fn, rowwise, id) {
  assert_shapes(shapes_in)
  assert_list(param_vals)
  assert_names(output_names, type = "unique")
  assert_function(fn)
  assert_flag(rowwise)
  assert_string(id)

  infer_shapes_once = function(shapes) {
    f = function(shapes, na_repl) {
      shape_orig = shapes
      if (rowwise) {
        shapes = shapes[-1L]
      }
      shapes[is.na(shapes)] = na_repl
      fn_args = names(formals(fn))
      filtered_params = param_vals[intersect(names(param_vals), fn_args)]

      # The "meta" device allocates nothing, so tracing costs no memory however large the shape is.
      # Not every operator implements it, but one that does not raises rather than returning a wrong
      # shape, so a failure there is retried on a real tensor instead of being reported: only the
      # retry's error says anything about the shape.
      trace_on = function(device) {
        tensor_in = mlr3misc::invoke(torch_empty, .args = shapes, device = torch_device(device))
        invoke(fn, tensor_in, .args = filtered_params)
      }
      tensor_out = tryCatch(trace_on("meta"), error = function(e) {
        tryCatch(trace_on("cpu"), error = function(e) {
          stopf("Input shape '%s' is invalid for PipeOp with id '%s' (unknown dimensions were replaced with %i): %s", # nolint
            shape_to_str(shape_orig), id, na_repl, conditionMessage(e))
        })
      })
      dim(tensor_out)
    }

    # A trace that fails is dropped: the smallest of the filled-in values exists to detect
    # operators that clamp to the input size, and operators that need a larger extent must not be
    # rejected because of it. Two traces are the minimum needed to tell a dimension that varies
    # with the input from one that does not.
    traced = lapply(na_replacements(if (rowwise) shapes[-1L] else shapes), function(na_repl) {
      tryCatch(list(shape = f(shapes, na_repl)), error = function(e) list(condition = e))
    })
    shapes_out = map(Filter(function(x) !is.null(x$shape), traced), "shape")
    if (length(shapes_out) < 2L) {
      # the traces with the larger values are the informative ones, so report the last failure
      condition = last(Filter(function(x) !is.null(x$condition), traced))$condition
      stopf("%s\nThe output shapes could not be inferred by tracing the operator, specify them explicitly instead (see the `shapes_out` argument).", conditionMessage(condition)) # nolint
    }

    if (length(unique(lengths(shapes_out))) > 1L) {
      stopf("Failed to infer shapes for PipeOp with id '%s', as the number of dimensions varies with different values filled in for the unknown dimensions.", id) # nolint
    }
    shapes_out = apply(do.call(rbind, shapes_out), 2, function(xs) {
      if (length(unique(xs)) == 1L) {
        return(xs[[1L]])
      }
      return(NA)
    })

    if (rowwise) {
      shapes_out = c(shapes[[1L]], shapes_out)
    }
    as.integer(shapes_out)
  }

  set_names(lapply(shapes_in, infer_shapes_once), output_names)
}
