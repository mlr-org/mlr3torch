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
#' @param only_batch_unknown (`logical(1)`)\cr
#'   Whether only the batch dimension can be `NA` in the input shapes or whether other
#'   dimensions can also be unknown.
#' @noRd
assert_shape = function(shape, null_ok = FALSE, coerce = TRUE, unknown_batch = NULL, len = NULL, only_batch_unknown = FALSE) { # nolint
  result = check_shape(shape, null_ok = null_ok, unknown_batch = unknown_batch, len = len, only_batch_unknown = only_batch_unknown) # nolint

  if (!isTRUE(result)) stopf(result)

  if (coerce && !is.null(shape)) {
    return(as.integer(shape))
  }
  shape
}


test_shape = function(shape, null_ok = FALSE, unknown_batch = NULL, len = NULL, only_batch_unknown = FALSE) {
  if (is.null(shape) && null_ok) {
    return(TRUE)
  }
  ok = test_integerish(shape, min.len = 1L, any.missing = TRUE, len = len)

  if (!ok) {
    return(FALSE)
  }

  if (only_batch_unknown && anyNA(shape[-1L])) {
    return(FALSE)
  }

  if (is.null(unknown_batch)) {
    # first dim can be present or missing
    return(TRUE)
  }
  return(is.na(shape[1L]) == unknown_batch)
}

check_shape = function(x, null_ok = FALSE, unknown_batch = NULL, len = NULL, only_batch_unknown = FALSE) {
  if (test_shape(x, null_ok = null_ok, unknown_batch = unknown_batch, len = len, only_batch_unknown = only_batch_unknown)) { # nolint
    return(TRUE)
  }
  sprintf("Invalid shape: %s.", shape_to_str(x))
}

assert_shapes = function(shapes, coerce = TRUE, named = FALSE, null_ok = FALSE, unknown_batch = NULL, only_batch_unknown = FALSE) { # nolint
  ok = test_list(shapes, min.len = 1L)
  if (named) {
    assert_names(setdiff(names(shapes), "..."), type = "unique")
  }
  if (!ok) {
    stopf("Invalid shape")
  }
  map(shapes, assert_shape, coerce = coerce, null_ok = null_ok, unknown_batch = unknown_batch, only_batch_unknown = only_batch_unknown) # nolint
}

check_rgb_shape = function(shape) {
  msg = check_shape(shape, len = 4L, null_ok = FALSE)
  if (!isTRUE(msg)) {
    return(msg)
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
assert_grayscale_or_rgb = function(shape) {
  assert_shape(shape, len = 4L, null_ok = FALSE, only_batch_unknown = TRUE)
  assert_true(shape[2L] == 3L || shape[2L] == 1L)
}

#' @title Infer Shapes
#' @description
#' Infer the shapes of the output of a function based on the shapes of the input.
#' This is done as follows:
#' 1. All `NA`s are replaced with values `1`, `2`, `3`.
#' 2. Three tensors are generated for the three shapes of step 1.
#' 3. The function is called on these three tensors and the shapes are calculated.
#' 4. If:
#'    * the number of dimensions varies, an error is thrown.
#'    * the number of dimensions is the same, values are set to `NA` if the dimension is varying
#'      between the three tensors and otherwise set to the unique value.
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
      if (rowwise) {
        shapes = shapes[-1L]
      }
      shapes[is.na(shapes)] = na_repl
      tensor_in = mlr3misc::invoke(torch_empty, .args = shapes, device = torch_device("cpu"))

      fn_args = names(formals(fn))
      filtered_params = param_vals[intersect(names(param_vals), fn_args)]

      tensor_out = tryCatch(invoke(fn, tensor_in, .args = filtered_params),
        error = function(e) {
          stopf("Input shape '%s' is invalid for PipeOp with id '%s'.", shape_to_str(list(sin)), id)
        }
      )
      dim(tensor_out)
    }

    shapes_out = lapply(1:3, f, shapes = shapes)

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
