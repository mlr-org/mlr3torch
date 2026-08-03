#' @title Check for Shape
#'
#' @description Checks whether an integer vector is a valid shape.
#' Unknown shapes are represted as `NULL`.
#'
#' @param shape (`integer()`)\cr
#' @param null_ok (`logical(1)`)\cr
#'   Whether `NULL` is a valid shape.
#' @param coerce (`logical(1)`)\cr
#'   Whether to coerce the input to an `integer()` if possible.
#' @param unknown_batch (`logical(1)`)\cr
#'   Whether the batch **must** be unknonw, i.e. `NA`.
#'   If left `NULL` (default), the first dimension can be `NA` or not.
#' @param len (`integer(1)`)\cr
#'   The length of the shape.
#' @noRd
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
  sprintf("Invalid shape: %s.", shape_to_str(x))
}

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

# Rejects an unknown (`NA`) size in the dimensions that a PipeOp reads when constructing its
# module: a convolution needs the number of input channels, but not the spatial extent.
# Without this, `NA_integer_` reaches libtorch, which fails with an unreadable C++ error.
# @param shape (`integer()`) The input shape, `NA` where a dimension is unknown.
# @param dims (`integer()`) Indices of the dimensions that must be known.
# @param what (`character(1)`) Describes those dimensions in the error message.
# @param id (`character(1)` | `NULL`) The PipeOp's id, `NULL` when called outside a PipeOp.
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

# Dimension-wise result of concatenating `shapes`, where the concatenated dimension is left as `NA`
# for the caller to fill in.
# Unlike broadcasting, `torch_cat()` requires all other dimensions to be *equal*, so a known size
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

# Broadcasting rules of torch, generalized to shapes that may contain `NA` (unknown).
# Per dimension a known size != 1 wins; if all known sizes are 1 and some input is unknown, the
# result is unknown, because the unknown one may be > 1 and would then determine the size.
# @param shapes (`list()` of `integer()`) The input shapes. They must already have the same number
#   of dimensions: we do not left-pad shorter shapes with 1s, because the first dimension
#   is the batch dimension.
# @param id (`character(1)`) The PipeOp's id, for the error message.
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

# Shape of `torch_reshape(x, shape)`. Like keras, the inferred dimension is resolved here whenever
# the number of input elements is known, and stays unknown otherwise.
# @param shape_in (`integer()`) The input shape.
# @param shape (`integer()`) The target shape, where `-1` (or `NA`) marks the dimension that torch
#   infers from the number of elements.
# @param id (`character(1)`) The PipeOp's id, for the error messages.
reshape_output_shape = function(shape_in, shape, id) {
  target = shape # the user's `shape`, for the error messages
  if (!length(shape)) {
    stopf("PipeOp '%s' requires 'shape' to have at least one dimension.", id)
  }
  shape[shape == -1] = NA
  if (any(!is.na(shape) & shape < 1)) {
    stopf("PipeOp '%s': 'shape' %s is invalid: every dimension must be at least 1, only -1 marks the dimension that is inferred from the number of elements.", # nolint
      id, shape_to_str(target))
  }
  unknown = which(is.na(shape))
  # torch can only infer one dimension, so more than one is invalid whatever the input shape is
  if (length(unknown) > 1L) {
    stopf("PipeOp '%s': 'shape' %s is invalid: at most one dimension can be inferred from the number of elements.", # nolint
      id, shape_to_str(target))
  }
  # the number of input elements is unknown as soon as one dimension is (typically the batch)
  inlen = prod(shape_in)
  if (is.na(inlen)) {
    return(as.integer(shape))
  }
  # note that `shape[-integer(0)]` is empty, so the two cases have to be distinguished
  knownlen = if (length(unknown)) prod(shape[-unknown]) else prod(shape)
  if (knownlen == 0 || (length(unknown) && inlen %% knownlen != 0) ||
      (!length(unknown) && inlen != knownlen)) {
    stopf("PipeOp '%s': 'shape' %s is not compatible with the input shape %s.",
      id, shape_to_str(target), shape_to_str(shape_in))
  }
  if (length(unknown)) {
    shape[unknown] = inlen / knownlen
  }
  # The first dimension is the batch dimension. A reshape that moves elements across it changes
  # the batch size, which fails much later with a mismatch against the target.
  rest_in = prod(shape_in[-1L])
  rest_out = prod(shape[-1L])
  keeps_batch = is.na(target[[1L]]) || target[[1L]] == -1
  if (keeps_batch && !is.na(rest_in) && !is.na(rest_out) && rest_in != rest_out) {
    stopf("PipeOp '%s': 'shape' %s changes the batch dimension of the input shape %s: it maps %s elements per observation to %s.", # nolint
      id, shape_to_str(target), shape_to_str(shape_in), rest_in, rest_out)
  }
  as.integer(shape)
}

# Resolves dimension indices that count from the end, as in torch, to positive ones.
# Indices that are out of range stay out of range, so that the assertions below report them.
# @param dim (`integer()`) The dimensions, negative ones counting back from the last: `-1` is the
#   last dimension, `-2` the one before it, and so on.
# @param shape (`integer()`) The shape that `dim` addresses.
# @param insert (`logical(1)`) Whether a dimension is inserted rather than addressed, as by
#   `nn_unsqueeze()`: there is then one more position than the shape has dimensions, `-1` appending
#   a new last one.
resolve_dim = function(dim, shape, insert = FALSE) {
  negative = dim < 0
  dim[negative] = length(shape) + as.integer(insert) + 1L + dim[negative]
  dim
}

# Rejects a `dim` parameter that does not address a dimension of `shape`.
# Without this check, assigning to an out-of-range index silently *extends* a shape with `NA`s
# instead of erroring, and a graph is then built on a shape that no tensor can have.
# @param dim (`integer(1)`) What the user specified, which may count down from the last dimension.
#   Only used for the error message, so that it reports the value the user knows.
# @param true_dim (`integer(1)`) The resolved index, see `resolve_dim()`.
# @param shape (`integer()`) The shape that `true_dim` addresses.
# @param id (`character(1)`) The PipeOp's id, for the error message.
assert_dim_in_range = function(dim, true_dim, shape, id) {
  if (true_dim >= 1L && true_dim <= length(shape)) {
    return(invisible(true_dim))
  }
  stopf("PipeOp '%s' cannot use 'dim' %i for the input shape %s, which has %i dimension(s).",
    id, dim, shape_to_str(shape), length(shape))
}

# Rejects an operation on the first dimension, which is the batch dimension.
# Changing it builds a network that only fails when the output no longer matches the target.
# @param dim (`integer(1)`) The resolved dimension that the operator changes.
# @param shape (`integer()`) The input shape, for the error message.
# @param id (`character(1)`) The PipeOp's id, for the error message.
assert_not_batch_dim = function(dim, shape, id) {
  if (dim != 1L) {
    return(invisible(dim))
  }
  stopf("PipeOp '%s' would change dimension 1 of the input shape %s, which is the batch dimension.",
    id, shape_to_str(shape))
}

# Halves one dimension of a shape, as the gated linear units do.
# An unknown dimension stays unknown, a known odd one is rejected.
# @param shape (`integer()`) The input shape.
# @param dim (`integer(1)`) The resolved dimension that is halved.
# @param id (`character(1)`) The PipeOp's id, for the error message.
halve_dim = function(shape, dim, id) {
  halved = shape[dim] / 2
  if (!test_integerish(halved)) {
    stopf("PipeOp '%s' requires dimension %i of the input shape %s to be divisible by 2, but it is %i.", # nolint
      id, dim, shape_to_str(shape), shape[dim])
  }
  shape[dim] = halved
  as.integer(shape)
}

# Rejects a shape with the wrong number of dimensions.
# The first dimension of a shape is always the batch dimension, so an operator over `d` dimensions
# needs a fixed number of them. Operators that read a specific dimension (a convolution reads the
# channel dimension) rely on this, because otherwise the dimension they read is a different one.
# @param shape (`integer()`) The input shape.
# @param ndim (`integer(1)`) The required number of dimensions, the batch dimension included.
# @param id (`character(1)`) The PipeOp's id, for the error message.
assert_ndim = function(shape, ndim, id) {
  if (length(shape) == ndim) {
    return(invisible(shape))
  }
  stopf("PipeOp '%s' requires an input with %i dimensions (the first one being the batch dimension), but got the shape %s, which has %i.", # nolint
    id, ndim, shape_to_str(shape), length(shape))
}

# Rejects an output extent that no tensor can have.
# A kernel that does not fit into the input, or a stride of 0, gives an output extent that is not
# positive (or not even finite). torch rejects such a configuration, so the shape must not be
# passed on to the rest of the graph, where it would produce follow-up errors far from the cause.
# @param extent (`numeric()`) The computed output extents. `NA` (unknown) ones cannot be proven
#   wrong and are accepted.
# @param shape_in (`integer()`) The input shape, for the error message.
# @param id (`character(1)` | `NULL`) The PipeOp's id, `NULL` when called outside a PipeOp.
assert_positive_extent = function(extent, shape_in, id) {
  invalid = !is.na(extent) & (!is.finite(extent) | extent < 1)
  if (!any(invalid)) {
    return(invisible(extent))
  }
  stopf("%s cannot be applied to the input shape %s: the output would have the size %s. Check 'kernel_size', 'stride', 'padding' and 'dilation'.", # nolint
    if (is.null(id)) "The operator" else sprintf("PipeOp '%s'", id), shape_to_str(shape_in),
    paste0(extent, collapse = ", "))
}

# Rejects inputs that do not all have the same number of dimensions, which the operators that
# combine several inputs require: we do not left-pad shorter shapes with 1s (batch dim)
# @param shapes (`list()` of `integer()`) The input shapes.
# @param id (`character(1)`) The PipeOp's id, for the error message.
assert_same_ndim = function(shapes, id) {
  ndim = map_int(shapes, length)
  if (length(unique(ndim)) == 1L) {
    return(invisible(shapes))
  }
  stopf("PipeOp '%s' requires all its inputs to have the same number of dimensions, but got the shapes %s (with %s dimensions).", # nolint
    id, shape_to_str(shapes), paste0(ndim, collapse = ", "))
}

check_rgb_shape = function(shape) {
  msg = check_shape(shape, len = 4L, null_ok = FALSE)
  if (!isTRUE(msg)) {
    return(msg)
  }
  if (is.na(shape[2L])) {
    return("Second dimension (the number of channels) must be known.")
  }
  if (shape[2L] != 3L) {
    return("Second dimension must be 3 for RGB images.")
  }
  return(TRUE)
}

assert_rgb_shape = function(shape) {
  msg = check_rgb_shape(shape)
  if (!isTRUE(msg)) {
    stopf(msg)
  }
  shape
}

# grayscale or rgb image
# only the channel dimension must be known, the spatial extent may be unknown
assert_grayscale_or_rgb = function(shape) {
  assert_shape(shape, len = 4L, null_ok = FALSE)
  assert_known_dims(shape, 2L, "the channel dimension (dimension 2)")
  assert_true(shape[2L] == 3L || shape[2L] == 1L,
    .var.name = "Second dimension is 3 for RGB images or 1 for grayscale images")
}

# The values that `infer_shapes()` fills in for the unknown dimensions before tracing a shape
# through a function. They span a wide range on purpose, because both ends are needed:
#
# * Large values (keras traces with 83 and 89, see `compute_output_spec()` in its torch backend)
#   are needed because operators that require a minimum extent fail on small ones -- a convolution
#   with a large kernel, for example -- which would reject a shape that is valid at runtime.
# * A small value is needed because several operators *clamp* to the input size instead:
#   `x[, 1:32]` and `transform_crop(height = 16)` return the input extent when it is smaller than
#   the requested one. Their output is therefore genuinely unknown, and tracing with large values
#   only would report it as known -- the traced value would then disagree with the tensor that the
#   network sees at runtime.
#
# None of the values may be 1: a dimension of size 1 broadcasts against everything and is squeezed
# away by operators such as `torch_squeeze()`, which changes the *number* of output dimensions.
# A trace that fails is dropped by `infer_shapes()`, so the small value costs nothing for operators
# that cannot handle it.
# @param shape (`integer()`) The shape whose `NA`s are replaced; only the number of unknown
#   dimensions and the size of the known ones matter.
# @param max_elements (`numeric(1)`) The number of elements the traced tensor may have. The
#   returned values are lowered to stay below it, which matters when many dimensions are unknown.
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
  last(candidates)
}

#' @title Infer Shapes
#' @description
#' Infer the shapes of the output of a function based on the shapes of the input.
#'
#' This is a heuristic and is only used for operators that the *user* supplies, i.e. [`nn_fn`][mlr3torch::mlr_pipeops_nn_fn]
#' without a `shapes_out` argument and [`pipeop_preproc_torch()`] with `shapes_out = "infer"`.
#' Every operator that `mlr3torch` itself provides computes its output shapes exactly.
#' Tracing cannot be exact: an operator whose output size is a step function of the input size can
#' return the same value for the traced inputs although it varies for others, in which case a
#' dimension is reported as known although it is not.
#' Specify the output shapes explicitly if this matters.
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
      tensor_in = mlr3misc::invoke(torch_empty, .args = shapes, device = torch_device("cpu"))

      fn_args = names(formals(fn))
      filtered_params = param_vals[intersect(names(param_vals), fn_args)]

      tensor_out = tryCatch(invoke(fn, tensor_in, .args = filtered_params),
        error = function(e) {
          stopf("Input shape '%s' is invalid for PipeOp with id '%s' (unknown dimensions were replaced with %i): %s", # nolint
            shape_to_str(shape_orig), id, na_repl, conditionMessage(e))
        }
      )
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
