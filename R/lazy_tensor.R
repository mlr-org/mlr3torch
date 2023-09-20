#' @title Create a lazy tensor
#' @description
#' Create a vector of class `"lazy_tensor"`, which is built upon the [`DataDescriptor`] class.
#' Each element of a lazy tensor contains an `id` and a [`DataDescriptor`] and represents
#' the output of the data descriptor's preprocessing graph applied to the element `id` of the data descriptor's
#' dataset.
#' Different elements can not only different `id`s but also different data descriptor.
  #' The only assumption is (currently) that they all have the same shape.
#'
#' In most cases using [`as_lazy_tensor()`] is more convenient.
#'
#' @param x (`list()` of `list()`s)\cr
#'   A list of lists, eaching having two elements `id` and `data_descriptor`.
#'   Usually the `id`s are simply the integers `1, ..., n` where `n` is the length of the dataset described by
#'   `data_descriptor`.
#' @param shape (`integer()`)\cr
#'   The shape of the lazy tensor. First element must be `NA` (the batch dimension).
#' @include DataDescriptor.R
#' @seealso lazy_tensor as_lazy_tensor, is_lazy_tensor
#' @export
#' @examples
#' # Create a dataset
#' dsg = dataset(
#'   initialize = function() self$x = torch_randn(10, 3, 3),
#'   .getitem = function(i) self$x[i, ],
#'   .length = function() nrow(self$x)
#' )
#' ds = dsg()
#'
#' dd = DataDescriptor(ds, list(x = c(NA, 3, 3)))
#'
#' x = map(seq_along(ds), function(i) list(id = i, data_descriptor = dd))
#'
#' lt = lazy_tensor(x, shape = c(NA, 3, 3))
#'
#' # the shape
#' lt$shape
#'
#' # More convenient ways for construction.
#' as_lazy_tensor(dd)
#' as_lazy_tensor(ds, list(x = c(NA, 3, 3)))
#' as_lazy_tensor(ds$x)
lazy_tensor = function(x = list(), shape = NULL) { # nolint
  assert_list(x)
  assert_shape(shape, null.ok = TRUE)

  if (length(x) == 0L) {
    return(new_lazy_tensor(list(), shape))
  }

  walk(x, function(elt) {
    assert_true(isTRUE(all.equal(names(elt), c("id", "data_descriptor"))))
  })

  shape_found = unique(map(x, function(elt) elt$data_descriptor$.pointer_shape))
  if (length(shape_found) > 1L) {
    stopf("All tensors must have the same shape.")
  }

  new_lazy_tensor(x, shape)
}

new_lazy_tensor = function(x, shape) {
  vctrs::new_vctr(x, shape = shape, class = "lazy_tensor")
}


#' @title Convert to lazy tensor
#' @description
#' Convert an object to type [`lazy_tensor`].
#' @param x (any)\cr
#'   Object to convert.
#' @param ... (any)\cr
#'   Additional parameters.
#' @export
as_lazy_tensor = function(x, ...) {
  UseMethod("as_lazy_tensor")
}


#' @export
as_lazy_tensor.NULL = function(x) { # nolint
  lazy_tensor()
}

#' @export
as_lazy_tensor.list = function(x, dataset_shapes) { # nolint
  lazy_tensor(x, dataset_shapes = dataset_shapes)
}

#' @export
as_lazy_tensor.torch_tensor = function(x, ...) { # nolint
  dsg = dataset(
    initialize = function(tensor) {
      self$tensor = tensor
    },
    .getbatch = function(ids) {
      list(tensor = self$tensor[ids, .., drop = FALSE]) # nolint
    },
    .length = function() nrow(self$tensor)
  )
  ds = dsg(x)
  shape = x$shape
  shape[1L] = NA_integer_
  as_lazy_tensor(DataDescriptor(dataset = ds, dataset_shapes = list(tensor = shape)), ...)
}

#' @param dataset_shapes (named `list()`)\cr
#'   The shapes of the output.
#'   Names are the elements of the list returned by the dataset.
#' @export
as_lazy_tensor.dataset = function(x, dataset_shapes, ...) { # nolint
  as_lazy_tensor(DataDescriptor(x, dataset_shapes = dataset_shapes, ...))
}

#' @param ids (`vector()`)\cr
#'   The ids that can be passed to the dataset.
#'   Defaults to `1, ..., length_dataset`.
#' @export
as_lazy_tensor.DataDescriptor = function(x, ids = NULL) { # nolint
  assert_class(x, "DataDescriptor")
  if (is.null(ids)) {
    ids = seq_along(x$dataset)
  } else {
    assert_vector(ids, length = length(x$dataset))
  }
  vec = map(ids, function(id) list(id = id, data_descriptor = x))
  if (length(x$dataset) == 0L) {
    return(lazy_tensor())
  }

  new_lazy_tensor(vec, shape = x$.pointer_shape)
}


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

  # In most practical cases we expect one DataDescriptor in a lazy tensor column.
  # We don't want to duplicate the DataDescriptor so we are using a hashmap to identify unique
  # DataDescriptors and add the PipeOp there.
  cache_env = new.env()
  walk(lt, function(elt) {
    if (!exists(elt$data_descriptor$.hash, cache_env)) {
      data_descriptor = elt$data_descriptor

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

      cache_env[[elt$data_descriptor$.hash]] = data_descriptor
    }
  })

  x = map(lt, function(x) {
    list(id = x$id, data_descriptor = cache_env[[x$data_descriptor$.hash]])
  })

  # TODO Use new_lazy_tensor for speedup (no shape checking)
  new_lazy_tensor(x, shape = shape)
}

merge_lazy_tensors = function(lts, pipeop, shape, clone_graph = TRUE) {
  .NotYetImplemented("merge_lazy_tensors")
}

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

#' @title Check for lazy tensor
#' @description
#' Checks whether an object is a lazy tensor.
#' @param x (any)\cr
#'   Object to check.
#' @export
is_lazy_tensor = function(x) {
  inherits(x, "lazy_tensor")
}


#' @export
vec_ptype2.lazy_tensor.list = function(x, y, ...) { # nolint
  list()
}

#' @export
vec_ptype2.list.lazy_tensor = function(x, y, ...) { # nolint
  list()
}

#' @export
vec_ptype2.lazy_tensor.lazy_tensor = function(x, y, ...) { # nolint
  if (is.null(x$shape) || is.null(y$shape) || isTRUE(all.equal(x$shape, y$shape))) {
    return(lazy_tensor(shape = x$shape %??% y$shape))
  }
  stopf("Shapes of lazy tensors must be equal.")
}

#' @export
vec_cast.lazy_tensor.lazy_tensor = function(x, to, ...) { # nolint
  if (length(to) == 0L || isTRUE(all.equal(x$shape, to$shape))) {
    return(x)
  }
  stopf("Cannot cast lazy tensor with shape '%s' to shape '%s'.",
    paste0(x$shape[-1L], collapse = "x"), paste0(to$shape[-1L], collapse = "x")
  )
}

#' @export
vec_cast.list.lazy_tensor = function(x, to, ...) { # nolint
  vec::vec_data(x)
}

#' @export
vec_cast.lazy_tensor.list = function(x, to, ...) { # nolint
  lazy_tensor(x)
}

#' @export
`[.lazy_tensor` = function(x, i, ...) { # nolint
  x = NextMethod()
  if (length(x) == 0L) {
    attr(x, "shape") = NULL
  }
  return(x)
}

#' @export
`$.lazy_tensor` = function(x, name) { # nolint
  if (isTRUE(name == "shape")) {
    return(attr(x, "shape"))
  }
  stopf("Lazy tensor does not have field '%s'.", name)
}

assert_lazy_tensor = function(x) {
  assert_class(x, "lazy_tensor")
}
