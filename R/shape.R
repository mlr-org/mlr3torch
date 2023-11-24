#' Check Shape
#'
#' Checks whether an integer vector is a valid shape.
#' Unknown shapes are represted as `NULL`.
#'
#' @param shape (`integer()`)\cr
#' @param null_ok (`logical(1)`)\cr
#'   Whether `NULL` is a valid shape.
#' @param coerce (`logical(1)`)\cr
#'   Whether to coerce the input to an `integer()` if possible.
assert_shape = function(shape, null_ok = FALSE, coerce = TRUE) {
  if (!test_shape(shape, null_ok = null_ok)) {
    stopf("Invalid shape: %s.", paste0(format(shape), collapse = ", "))
  }
  if (coerce && !is.null(shape)) {
    return(as.integer(shape))
  }
  shape
}

test_shape = function(shape, null_ok = FALSE) {
  if (is.null(shape) && null_ok) {
    return(TRUE)
  }
  ok = test_integerish(shape, min.len = 2L, all.missing = TRUE)

  if (!ok) {
    return(FALSE)
  }

  is_na = is.na(shape)
  if (anyNA(shape[-1L])) {
    return(FALSE)
  }
  return(TRUE)
}

check_shape = function(x, null_ok = FALSE) {
  if (test_shape(x, null_ok = null_ok)) {
    return(TRUE)
  }
  "Must be a valid shape."
}

assert_shapes = function(shapes, named = TRUE, null_ok = FALSE) { # nolint
  assert_list(shapes, names = if (named && !identical(unique(names(shapes)), "...")) "unique", min.len = 1L)
  map(shapes, assert_shape, null_ok = null_ok)
}
