#' @title Create an object of class "imageuri"
#' @description
#' Creates an object of class `"imageuri"`, that contains the uris of images.
#' @param obj (`character()`)\cr
#'   Character vector containing the paths to images.
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
  obj[...]
}

#' @export
`[[<-.imageuri` = function(obj, ..., value) {
  assert_character(value)
  obj = unclass(obj)
  obj[[...]] = value
  imageuri(obj)
}

#' @export
`[<-.imageuri` = function(obj,  ..., value) { # nolint
  # imageuri inherits from character
  assert_character(value)
  obj = unclass(obj)
  obj[...] = value
  imageuri(obj)
}

#' @export
c.imageuri = function(...) { # nolint
  dots = list(...)
  if (!all(map_lgl(dots, function(x) test_character(x)))) {
    stopf("To concatenate imageuri vectors, all objects must inherit from 'character'.")
  }
  imageuri(do.call(c, lapply(dots, unclass)))
}

assert_imageuri = function(obj) {
  assert_class(obj, c("imageuri", "list"))
}

#' @title Conver to imageuri
#' @description
#' Converts an object to class [`imageuri`].
#' @param obj (any)\cr
#'   Object to convert.
#' @param ... (any)\cr
#'   Additional arguments.
#' @return ([`imageuri`])\cr
as_imageuri = function(obj, ...) {
  UseMethod("as_imageuri")
}

#' @export
as_imageuri.imageuri = function(obj, ...) { # nolint
  obj
}

#' @export
as_imageuri.character = function(obj, ...) { # nolint
  imageuri(obj)
}
