#' @title Data Descriptor
#'
#' @description
#' A data descriptor is a rather internal data structure used in the [`lazy_tensor`] data type.
#' In essence it is an annotated [`torch::dataset`] and a preprocessing graph (consisting mosty of [`PipeOpModule`]
#' operators). The additional meta data (e.g. pointer, shapes) allows to preprocess [`lazy_tensors`] in an
#' [`mlr3pipelines::Graph`] just like any (non-lazy) data types.
#' To observe the effect of such a preprocessing, [`materialize()`] can be used.
#'
#' @param dataset ([`torch::dataset`])\cr
#'   The torch dataset.
#' @param dataset_shapes (named `list()` of (`integer()` or `NULL`))\cr
#'   The shapes of the output.
#'   Names are the elements of the list returned by the dataset.
#'   If the shape is not `NULL` (unknown, e.g. for images of different sizes) the first dimension must be `NA` to
#'   indicate the batch dimension.
#' @param graph ([`Graph`])\cr
#'  The preprocessing graph.
#'  If left `NULL`, no preprocessing is applied to the data and `.input_map`, `.pointer` and `.pointer_shape`
#'  are inferred in case the dataset returns only one element.
#' @param .input_map (`character()`)\cr
#'   Character vector that must have the same length as the input of the graph.
#'   Specifies how the data from the `dataset` is fed into the preprocessing graph.
#' @param .pointer (`character(2)` | `NULL`)\cr
#'   Indicating an element on which a model is. Points to an output channel within `graph`:
#'   Element 1 is the `PipeOp`'s id and element 2 is that `PipeOp`'s output channel.
#' @param .pointer_shape (`integer` | `NULL`)\cr
#'   Shape of the output indicated by `.pointer`.
#' @param clone_graph (`logical(1)`)\cr
#'   Whether to clone the preprocessing graph.
#' @param .pointer_shape_predict (`integer()` or `NULL`)\cr
#'   Internal use only.
#'   Used in a [`Graph`] to anticipate possible mismatches between train and predict shapes.
#'
#' @export
#' @seealso ModelDescriptor, lazy_tensor
#' @examples
#' # Create a dataset
#' dsg = dataset(
#'   initialize = function() self$x = torch_randn(10, 3, 3),
#'   .getitem = function(i) self$x[i, ],
#'   .length = function() nrow(self$x)
#' )
#' ds = dsg()
#'
#' # Create the preprocessing graph
#' po_module = po("module", module = function(x) torch_reshape(x, c(-1, 9)))
#' po_module$output
#' graph = as_graph(po_module)
#'
#' # Create the data descriptor
#'
#' dd = DataDescriptor(
#'   dataset = ds,
#'   dataset_shapes = list(x = c(NA, 3, 3)),
#'   graph = graph,
#'   .input_map = "x",
#'   .pointer = c("module", "output"),
#'   .pointer_shape = c(NA, 9)
#' )
#'
#' # with no preprocessing
#' dd1 = DataDescriptor(ds, list(x = c(NA, 3, 3)))
DataDescriptor = function(dataset, dataset_shapes, graph = NULL, .input_map = NULL, .pointer = NULL,
  .pointer_shape = NULL, clone_graph = TRUE, .pointer_shape_predict = NULL) {
  assert_class(dataset, "dataset")

  # If the dataset implements a .getbatch() method the shape must be specified.
  assert_shapes(dataset_shapes, null_ok = is.null(dataset$.getbatch))
  assert_shape(.pointer_shape_predict, null_ok = TRUE)

  if (is.null(graph)) {
    if ((length(dataset_shapes) == 1L) && is.null(.input_map)) {
      .input_map = names(dataset_shapes)
    }
    assert_true(length(.input_map) == 1L)
    assert_subset(.input_map, names(dataset_shapes))

    graph = as_graph(po("nop", id = paste0(class(dataset)[[1L]], "_", .input_map)))
    .pointer = c(graph$output$op.id, graph$output$channel.name)
    .pointer_shape = dataset_shapes[[.input_map]]
  } else {
    graph = as_graph(graph, clone = clone_graph)
    assert_true(length(graph$pipeops) >= 1L)

    if (any(is.null(.input_map), is.null(.pointer), is.null(.pointer_shape))) {
      stopf("When passing a graph you need to specify .input_map, .pointer and .pointer_shape.")
    }

    assert_choice(.pointer[[1]], names(graph$pipeops))
    assert_choice(.pointer[[2]], graph$pipeops[[.pointer[[1]]]]$output$name)
    assert_subset(paste0(.pointer, collapse = "."), graph$output$name)
    assert_shape(.pointer_shape, null_ok = TRUE)

    assert_subset(.input_map, names(dataset_shapes))
    assert_true(length(.input_map) == length(graph$input$name))
  }

  # We hash the address of the environment, because the hashes of an environment are not stable,
  # even with a .dataset (that should usually not really have a state), hashes might change due to byte-code
  # compilation
  dataset_hash = calculate_hash(address(dataset))

  obj = structure(
    list(
      dataset = dataset,
      graph = graph,
      dataset_shapes = dataset_shapes,
      .input_map = .input_map,
      .pointer = .pointer,
      .pointer_shape = .pointer_shape,
      .dataset_hash = dataset_hash,
      .hash = NULL, # is set below
      # Once a DataDescriptor is created the input PipeOps are fix, we save them
      # here because they can be costly to compute
      .graph_input = graph$input$name,
      .pointer_shape_predict = .pointer_shape_predict
    ),
    class = "DataDescriptor"
  )

  obj = set_data_descriptor_hash(obj)

  return(obj)
}

# TODO: printer

set_data_descriptor_hash = function(data_descriptor) {
  data_descriptor$.hash = calculate_hash(
    data_descriptor$.dataset_hash,
    data_descriptor$graph$hash,
    data_descriptor$.input_map,
    data_descriptor$.pointer,
    data_descriptor$.pointer_shape
  )
  return(data_descriptor)
}
#' @title Create a lazy tesornsor
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
  shape = x$data_descriptor$.pointer_shape
  shape = if (is.null(shape)) {
    return(rep("<tnsr[]>", length(x)))
  }
  shape = paste0(x$data_descriptor$.pointer_shape[-1L], collapse = "x")

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
#' identical(lt$graph, lt_plus_five$graph)
#' lt$graph$edges
#' lt_plus_five$graph_edges
#' # pipeops are not cloned
#' identical(lt$graph$pipeops[[1]], lt_plus_five$graph[[1]])
#' @noRd
transform_lazy_tensor = function(lt, pipeop, shape, shape_predict = NULL) {
  assert_lazy_tensor(lt)
  assert_class(pipeop, "PipeOpModule")
  assert_true(nrow(pipeop$input) == 1L)
  assert_true(nrow(pipeop$output) == 1L)
  assert_shape(shape, null_ok = TRUE)
  # shape_predict can be NULL if we transform a tensor during `$predict()` in PipeOpTaskPreprocTorch
  assert_shape(shape_predict, null_ok = TRUE)

  data_descriptor = lt$data_descriptor

  data_descriptor$graph = data_descriptor$graph$clone(deep = FALSE)
  data_descriptor$graph$edges = copy(data_descriptor$graph$edges)

  data_descriptor$graph$add_pipeop(pipeop, clone = FALSE)
  data_descriptor$graph$add_edge(
    src_id = data_descriptor$.pointer[1L],
    src_channel = data_descriptor$.pointer[2L],
    dst_id = pipeop$id,
    dst_channel = pipeop$input$name
  )

  data_descriptor$.pointer = c(pipeop$id, pipeop$output$name)
  data_descriptor$.pointer_shape = shape
  data_descriptor$.pointer_shape_predict = shape_predict
  data_descriptor = set_data_descriptor_hash(data_descriptor)

  new_lazy_tensor(data_descriptor, map_int(vec_data(lt), 1))
}

#' @export
`$.lazy_tensor` = function(x, name) {
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
