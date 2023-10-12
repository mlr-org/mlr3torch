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

#' @title Transform Lazy Tensor
#' @description
#' Input must be PipeOpModule with exactly one input and one output
#' shape must be shape with NA in first dimension
#' @param lt ([`lazy_tensor`])\cr
#'   A lazy tensor vector.
#' @param pipeop ([`PipeOpModule`])\cr
#'   The pipeop to be added to the preprocessing graph(s) of the lazy tensor.
#'   Must have one input and one output.
#' @param shape (`integer()`)\cr
#'   The shape of the lazy tensor (without the batch dimension).
#' @param clone_graph (`logical(1)`)\cr
#'   Whether to clone the graph from the data descriptor.
#' @noRd
transform_lazy_tensor = function(lt, pipeop, shape, clone_graph = TRUE) {
  assert_lazy_tensor(lt)
  assert_class(pipeop, "PipeOpModule")
  assert_true(nrow(pipeop$input) == 1L)
  assert_true(nrow(pipeop$output) == 1L)
  assert_shape(shape)
  assert_flag(clone_graph)

  data_descriptor = attr(lt, "data_descriptor")

  if (clone_graph) {
    data_descriptor$graph = data_descriptor$graph$clone(deep = TRUE)
  }

  data_descriptor$graph$add_pipeop(pipeop$clone(deep = TRUE))
  data_descriptor$graph$add_edge(
    src_id = data_descriptor$.pointer[1L],
    src_channel = data_descriptor$.pointer[2L],
    dst_id = pipeop$id,
    dst_channel = pipeop$input$name
  )

  data_descriptor$.pointer = c(pipeop$id, pipeop$output$name)
  data_descriptor$.pointer_shape = shape
  data_descriptor = set_data_descriptor_hash(data_descriptor)

  new_lazy_tensor(data_descriptor, vec_data(lt))
}
