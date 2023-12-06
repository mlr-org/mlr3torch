#'
#' Checks whether an integer vector is a valid shape.
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
assert_shape = function(shape, null_ok = FALSE, coerce = TRUE, unknown_batch = NULL) {
  result = check_shape(shape, null_ok = null_ok, unknown_batch = unknown_batch)

  if (!isTRUE(result)) stopf(result)

  if (coerce && !is.null(shape)) {
    return(as.integer(shape))
  }
  shape
}


#FIX SHAPE IN FIRST DIM:
# * should not happen for DataDescriptor (subsetting)
# * should not be possible in ModelDescriptor (makes no sense)
# * still, shapes_out() should also handle this case

test_shape = function(shape, null_ok = FALSE, unknown_batch = NULL) {
  if (is.null(shape) && null_ok) {
    return(TRUE)
  }
  ok = test_integerish(shape, min.len = 2L, all.missing = TRUE)

  if (!ok) {
    return(FALSE)
  }

  if (anyNA(shape[-1L])) {
    return(FALSE)
  }
  if (is.null(unknown_batch)) {
    # first dim can be present or missing
    return(TRUE)
  }
  return(is.na(shape[1L]) == unknown_batch)
}

check_shape = function(shape, null_ok = FALSE, unknown_batch = NULL) {
  if (test_shape(shape, null_ok = null_ok, unknown_batch = unknown_batch)) {
    return(TRUE)
  }
  stopf("Invalid shape: %s.", paste0(format(shape), collapse = ", "))
}
assert_shapes = function(shapes, coerce = TRUE, named = FALSE, null_ok = FALSE, unknown_batch = NULL) { # nolint
  ok = test_list(shapes, names = if (named && !identical(unique(names(shapes)), "...")) "unique", min.len = 1L)
  if (!ok) {
    stopf("Invalid shape")
  }
  map(shapes, assert_shape, coerce = coerce, null_ok = null_ok, unknown_batch = unknown_batch)
}