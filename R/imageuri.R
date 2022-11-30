#' @title Create an object of class "imageuri"
#' @description
#' Creates an object of class `"imageuri"`, that contains the uris of images.
#' @param x (`character()`)\cr
#'   Character vector containig the paths to images.
#' @export
imageuri = function(x) {
  assert_character(x)
  structure(
    x,
    class = c("imageuri", "character")
  )
}

#' @export
`[.imageuri` = function(obj, ...) {
  imageuri(unclass(obj)[...])
}
