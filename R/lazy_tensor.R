#' @title Create a lazy tensor
#'
#' @description
#' Create a lazy tensor.
#' @param data_descriptor ([`DataDescriptor`] or `NULL`)\cr
#'   The data descriptor or `NULL` for a lazy tensor of length 0.
#' @param ids (`integer()`)\cr
#'   The elements of the `data_descriptor` that the created lazy tensor contains.
#' @export
lazy_tensor = function(data_descriptor = NULL, ids = NULL) {
  assert_class(data_descriptor, "DataDescriptor", null.ok = TRUE)
  if (is.null(data_descriptor)) {
    assert_integerish(ids, len = 0L, null.ok = TRUE)
    return(new_lazy_tensor(NULL, integer(0)))
  }
  if (is.null(ids)) {
    ids = seq_along(data_descriptor$dataset)
  } else {
    assert_integerish(ids, lower = 1L, upper = length(data_descriptor$dataset), any.missing = FALSE)
  }

  new_lazy_tensor(data_descriptor, ids)
}

new_lazy_tensor = function(data_descriptor, ids) {
  vctrs::new_vctr(ids, data_descriptor = data_descriptor, class = "lazy_tensor")
}

#' @export
format.lazy_tensor = function(x, ...) { # nolint
  if (!length(x)) return(character(0))
  shape = paste0(attr(x, "data_descriptor")$.pointer_shape[-1L], collapse = "x")
  map_chr(x, function(elt) {
    sprintf("<tnsr[%s]>", shape)
  })
}


#' @title Convert to lazy tensor
#' @description
#' Convert a object to a [`lazy_tensor()`].
#' @param x (any)\cr
#'   Object to convert to a [`lazy_tensor()`]
#' @param ... (any)\cr
#'  Additional arguments passed to the method.
#' @export
as_lazy_tensor = function(x, ...) {
  UseMethod("as_lazy_tensor")
}

#' @export
as_lazy_tensor.DataDescriptor = function(x, ids = NULL) { # nolint
  lazy_tensor(x, ids = ids)
}

#' @export
as_lazy_tensor.dataset = function(x, dataset_shapes, ids = NULL, ...) { # nolint
  dd = DataDescriptor(dataset = x, dataset_shapes = dataset_shapes, ...)
  lazy_tensor(dd, ids)
}

#' @export
as_lazy_tensor.torch_tensor = function(x) { # nolint
  ds = dataset(
    initialize = function(x) {
      self$x = x
    },
    .getbatch = function(ids) {
      list(x = self$x[ids, .., drop = FALSE]) # nolint
    },
    .length = function(ids) {
      dim(self$x)[1L]
    }
  )(x)
  as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, dim(x))))
}


#' @export
vec_ptype_abbr.lazy_tensor <- function(x, ...) { # nolint
  "ltnsr"
}

#' @export
`[.lazy_tensor` = function(x, i, ...) { # nolint
  x = NextMethod()
  if (length(x) == 0L) {
    attr(x, "data_descriptor") = NULL
  }
  return(x)
}

#' @title Check for lazy tensor
#' @description
#' Checks whether an object is a lazy tensor.
#' @param x (any)\cr
#'   Object to check.
#' @export
is_lazy_tensor = function(x) {
  inherits(x, "lazy_tensor")
}

