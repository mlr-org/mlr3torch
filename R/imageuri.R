#' @title Create an object of class "imageuri"
#' @description
#' Creates an object of class `"imageuri"`, that contains the uris of images.
#' @param obj (`character()`)\cr
#'   Character vector containig the paths to images.
#' @export
imageuri = function(obj) {
  # TODO: examples
  assert_character(obj)
  structure(
    obj,
    class = c("imageuri", "character")
  )
}

#' @export
`[.imageuri` = function(obj, ...) {
  imageuri(unclass(obj)[...])
}

#' @export
`[[.imageuri` = function(obj, ...) {
  imageuri(unclass(obj)[...])
}

#' @export
`[[<-.imageuri` = function(obj, ..., value) {
  assert_character(value)
  obj = unclass(obj)
  obj[[...]] = value
  imageuri(value)
}

#' @export
`[<-.imageuri` = function(obj,  ..., value) { # nolint
  assert_character(value)
  obj = unclass(obj)
  obj[...] = value
  imageuri(value)
}

#' @export
c.imageuri = function(...) { # nolint
  dots = list(...)
  if (!all(map_lgl(dots, function(x) test_character(x)))) {
    stopf("To concatenate an imageuri, all objects must inherit from 'character'.")
  }
  imageuri(do.call(c, lapply(dots, unclass)))
}

assert_imageuri = function(obj) {
  assert_class(obj, c("imageuri", "list"))
}

as_imageuri = function(obj) {
  if (test_class(obj, "imageuri")) {
    obj
  } else if (test_character(obj)) {
    imageuri(obj)
  } else {
    stopf("Cannot convert object of class '%s' to 'imageuri'", class(obj)[[1L]])
  }
}

# TODO: printer
