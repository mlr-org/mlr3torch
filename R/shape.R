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
  ok = test_integerish(shape, min.len = 2L, all.missing = FALSE, any.missing = TRUE, len = len)

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

check_shape = function(x, null_ok = FALSE, unknown_batch = NULL, len = NULL) {
  if (test_shape(x, null_ok = null_ok, unknown_batch = unknown_batch, len = len)) {
    return(TRUE)
  }
  sprintf("Invalid shape: %s.", paste0(format(x), collapse = ", "))
}

assert_shapes = function(shapes, coerce = TRUE, named = FALSE, null_ok = FALSE, unknown_batch = NULL) { # nolint
  ok = test_list(shapes, min.len = 1L)
  if (named) {
    assert_names(setdiff(names(shapes), "..."), type = "unique")
  }
  if (!ok) {
    stopf("Invalid shape")
  }
  map(shapes, assert_shape, coerce = coerce, null_ok = null_ok, unknown_batch = unknown_batch)
}

assert_rgb_shape = function(shape) {
  assert_shape(shape, len = 4L, null_ok = FALSE)
  assert_true(shape[2L] == 3L)
}

# grayscale or rgb image
assert_grayscale_or_rgb = function(shape) {
  assert_shape(shape, len = 4L, null_ok = FALSE)
  assert_true(shape[2L] == 3L || shape[2L] == 1L)
}
