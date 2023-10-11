#' @export
format.lazy_tensor = function(x, ...) { # nolint
  if (!length(x)) return(character(0))
  shape = paste0(x$shape[-1L], collapse = "x")
  map_chr(x, function(elt) {
    sprintf("<tnsr[%s]>", shape)
  })
}


#' @export
vec_ptype_abbr.lazy_tensor <- function(x, ...) { # nolint
  "ltnsr"
}
