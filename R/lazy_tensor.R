#' @title Create a lazy tesornsor
#'
#' @description
#' Create a lazy tensor.
#' @param data_descriptor ([`DataDescriptor`] or `NULL`)\cr
#'   The data descriptor or `NULL` for a lazy tensor of length 0.
#' @param ids (`integer()`)\cr
#'   The elements of the `data_descriptor` that the created lazy tensor contains.
#' @include DataDescriptor.R
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
  # previously, only the id was included and not the hash
  # this led to issues with stuff like unlist(), which dropped attribute and suddenly the lazy_tensor column
  # was a simple integer vector. (this caused stuff like PipeOpFeatureUnion to go havock and lead to bugs)
  # For this reason, we now also include the hash of the data_descriptor
  # We can then later also use this to support different DataDescriptors in a single lazy tensor column

  # Note that we just include the hash as an attribute, so c() does not allow to combine lazy tensors whose
  # data descriptors have different hashes.
  vctrs::new_vctr(map(ids, function(id) list(id, data_descriptor)), hash = data_descriptor$.hash, class = "lazy_tensor")
}

#' @export
format.lazy_tensor = function(x, ...) { # nolint
  if (!length(x)) return(character(0))
  shape = dd(x)$.pointer_shape
  shape = if (is.null(shape)) {
    return(rep("<tnsr[]>", length(x)))
  }
  shape = paste0(dd(x)$.pointer_shape[-1L], collapse = "x")

  map_chr(x, function(elt) {
    sprintf("<tnsr[%s]>", shape)
  })
}

dd = function(x) {
  if (!length(x)) {
    stopf("Cannot access data descriptor when lazy_tensor has length 0.")
  }
  x[[1L]][[2L]]
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
as_lazy_tensor.numeric = function(x) { # nolint
  as_lazy_tensor(torch_tensor(x))
}

#' @export
as_lazy_tensor.torch_tensor = function(x) { # nolint
  if (length(dim(x)) == 1L) {
    x = x$unsqueeze(2)
  }
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
  as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, dim(x)[-1])))
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
  test_class(x, "lazy_tensor")
}

#' @title Transform Lazy Tensor
#' @description
#' Transform  a [`lazy_tensor`] vector by appending a preprocessing step.
#'
#' @param lt ([`lazy_tensor`])\cr
#'   A lazy tensor vector.
#' @param pipeop ([`PipeOpModule`])\cr
#'   The pipeop to be added to the preprocessing graph(s) of the lazy tensor.
#'   Must have one input and one output.
#'   Is not cloned, so should be cloned beforehand.
#' @param shape (`integer()`)\cr
#'   The shape of the lazy tensor.
#' @param shape_predict (`integer()`)\cr
#'   The shape of the lazy tensor if it was applied during `$predict()`.
#'
#' @details
#' The following is done:
#' 1. A shallow copy of the [`lazy_tensor`]'s preprocessing `graph` is created.
#' 1. The provided `pipeop` is added to the (shallowly cloned) `graph` and connected to the current `.pointer` of the
#' [`DataDescriptor`].
#' 1. The `.pointer` of the [`DataDescriptor`] is updated to point to the new output channel of the `pipeop`.
#' 1. The `.pointer_shape` of the [`DataDescriptor`] set to the provided `shape`.
#' 1. The `.hash` of the [`DataDescriptor`] is updated.
#' Input must be PipeOpModule with exactly one input and one output
#' shape must be shape with NA in first dimension
#'
#' @return [`lazy_tensor`]
#' @examples
#' lt = as_lazy_tensor(1:10)
#' add_five = po("module", module = function(x) x + 5)
#' lt_plus_five = transform_lazy_tensor(lt, add_five, c(NA, 1))
#' torch_cat(list(materialize(lt, rbind = TRUE),  materialize(lt_plus_five, rbind = TRUE)), dim = 2)
#' # graph is cloned
#' identical(lt[[1]][[2]]$graph, lt_plus_five[[1]][[2]]$graph)
#' lt[[1]][[2]]$graph$edges
#' lt_plus_five[[1]][[2]]$graph$edges
#' # pipeops are not cloned
#' identical(lt[[1]][[2]]$graph$pipeops[[1]], lt_plus_five[[1]][[2]]$graph[[1]])
#' @noRd
transform_lazy_tensor = function(lt, pipeop, shape, shape_predict = NULL) {
  assert_lazy_tensor(lt)
  assert_class(pipeop, "PipeOpModule")
  assert_true(nrow(pipeop$input) == 1L)
  assert_true(nrow(pipeop$output) == 1L)
  assert_shape(shape, null_ok = TRUE, unknown_batch = TRUE)
  # shape_predict can be NULL if we transform a tensor during `$predict()` in PipeOpTaskPreprocTorch
  assert_shape(shape_predict, null_ok = TRUE, unknown_batch = TRUE)

  data_descriptor = dd(lt)

  graph = data_descriptor$graph$clone(deep = FALSE)
  graph$edges = copy(data_descriptor$graph$edges)

  graph$add_pipeop(pipeop, clone = FALSE)
  graph$add_edge(
    src_id = data_descriptor$.pointer[1L],
    src_channel = data_descriptor$.pointer[2L],
    dst_id = pipeop$id,
    dst_channel = pipeop$input$name
  )

  data_descriptor = DataDescriptor(
    data_descriptor$dataset,
    dataset_shapes = data_descriptor$dataset_shapes,
    graph = graph,
    .input_map = data_descriptor$.input_map,
    .pointer = c(pipeop$id, pipeop$output$name),
    .pointer_shape = shape,
    .pointer_shape_predict = shape_predict,
    clone_graph = FALSE
  )

  new_lazy_tensor(data_descriptor, map_int(vec_data(lt), 1))
}

#' @export
`$.lazy_tensor` = function(x, name) {
  # FIXME: remove this method
  #stop("Not supported anymore")
  if (!length(x)) {
    stop("lazy tensor has length 0.")
  }

  dd = x[[1L]][[2L]]
  if (name == "data_descriptor") {
    return(dd)
  }

  assert_choice(name, c(
    "dataset",
    "graph",
    "dataset_shapes",
    ".input_map",
    ".pointer",
    ".pointer_shape",
    ".dataset_hash",
    ".hash",
    ".graph_input",
    ".pointer_shape_predict"
  ))
  dd[[name]]
}