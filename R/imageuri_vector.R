#' @title Create an object of class "imageuri_vector"
#' @description
#' Creates an object of class `"imageuri_vector"`, that contains the uris of images.
#' @param obj (`character()`)\cr
#'   Character vector containig the paths to images.
#' @export
imageuri_vector = function(obj) {
  # TODO: examples
  assert_character(obj)
  structure(
    obj,
    class = c("imageuri_vector", "character")
  )
}

#' @export
`[.imageuri_vector` = function(obj, ...) {
  imageuri_vector(unclass(obj)[...])
}

#' @export
`[[.imageuri_vector` = function(obj, ...) {
  imageuri_vector(unclass(obj)[...])
}

#' @export
`[[<-.imageuri_vector` = function(obj, value, ...) {
  assert_character(value)
  obj = unclass(obj)
  obj[[...]] <- value
  imageuri_vector(value)
}

#' @export
`[<-.imageuri_vector` = function(obj, value, ...) { # nolint
  assert_character(value)
  obj = unclass(obj)
  obj[...] <- value
  imageuri_vector(value)
}

#' @export
c.imageuri_vector = function(...) { # nolint
  dots = list(...)
  if (!all(map_lgl(dots, function(x) test_character(x)))) {
    stopf("To concatenate an imageuri_vector, all objects must inherit from 'character'.")
  }
  imageuri_vector(do.call(c, lapply(dots, unclass)))
}

assert_imageuri_vector = function(obj) {
  assert_class(obj, c("imageuri_vector", "list"))
}

as_imageuri_vector = function(obj) {
  if (test_class(obj, "imageuri_vector")) {
    obj
  } else if (test_character(obj)) {
    imageuri_vector(obj)
  } else {
    stopf("Cannot convert object of class '%s' to 'imageuri_vector'", class(obj)[[1L]])
  }
}

# TODO: printer
