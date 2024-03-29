#' @title Create an object of class "imageuri"
#' @description
#'   Creates a class "images", that contains the uris of images.
#' @param x (`character()`)\cr
#'   Character vector containig the paths to the images.
#' @export
imageuri = function(x) {
  assert_character(x)
  # input = torchvision::transform_to_tensor(input)$to(device = "cpu")
  structure(
    x,
    class = c("imageuri", "character")
  )
}

#' @export
`[.imageuri` = function(obj, ...) {
  imageuri(unclass(obj)[...])
}

#' @title Transforms an Imageuri
#'
#' @description
#' Transforms an imageuri.
#'
#' @param x (`imageuri()`)\cr
#'   Imageuri vector.
#' @param trafo (`function`)\cr
#'   The image transformation. Must be a function with one argument.
#'
#' @export
transform_imageuri = function(x, trafo) {
  assert_true(class(x)[[1L]] == "imageuri")
  assert_function(trafo, nargs = 1L)
  transformed = attr(x, "transformed")
  transformed = trafo(transformed)
  # transformed = try(trafo(input), silent = TRUE)
  if (inherits(transformed, "try-error")) {
    stopf("Transformation invalid for example input.")
  }
  if (!inherits(transformed, "torch_tensor")) {
    stopf("Transformation did not produce an object of class 'torch_tensor' for the example input.")
  }
  attr(x, "trafos") = c(attr(x, "trafos"), trafo)
  attr(x, "transformed") = transformed
  return(x)
}

#' @export
print.imageuri = function(x, ...) {
  n_trafos = length(attr(x, "trafos"))
  catf("<imageuri>")
  catf(" * N(Trafos): %d", n_trafos)
  catf(" * Data Type: %s", class(attr(x, "transformed"))[[1L]])
  catf(" * Example uri: %s", x[[1L]])
  invisible(x)
}
