#' @title Data Descriptor
#'
#' @description
#' A data descriptor is a rather internal data structure used in the [`lazy_tensor`] data type.
#' In essence it is an annotated [`torch::dataset`] and a preprocessing graph (consisting mosty of [`PipeOpModule`]
#' operators). The additional meta data (e.g. pointer, shapes) allows to preprocess [`lazy_tensors`] in an
#' [`mlr3pipelines::Graph`] just like any (non-lazy) data types.
#' The preprocessing is applied when [`materialize()`] is called on the [`lazy_tensor`].
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
#' dd_no_preproc = DataDescriptor(ds, list(x = c(NA, 3, 3)))
#' dd_no_preproc
DataDescriptor = function(dataset, dataset_shapes, graph = NULL, .input_map = NULL, .pointer = NULL,
  .pointer_shape = NULL, .pointer_shape_predict = NULL, clone_graph = TRUE) {
  assert_class(dataset, "dataset")

  # If the dataset implements a .getbatch() method the shape must be specified, as it should be the same for
  # all batches
  # For simplicity we here require the first dimension of the shape to be NA so we don't have to deal with it,
  # e.g. during subsetting
  assert_shapes(dataset_shapes, null_ok = is.null(dataset$.getbatch), unknown_batch = TRUE, named = TRUE)
  assert_shape(.pointer_shape_predict, null_ok = TRUE, unknown_batch = TRUE)

  # not sure whether we should keep this?
  #example = if (is.null(dataset$.getbatch)) dataset$.getitem(1L) else dataset$.getbatch(1L)
  #if (!test_list(example, names = "unique") || !test_permutation(names(example), names(dataset_shapes))) {
  #  stopf("Dataset must return a list with named elements that are a permutation of the dataset_shapes names.")
  #  iwalk(dataset_shapes, function(dataset_shape, name) {
  #    if (!is.null(dataset_shape) && !test_equal(dataset_shapes[[name]][-1], example[[name]]$shape[-1L])) {
  #      stopf("First batch from dataset is incompatible with the provided dataset_shapes.")
  #    }
  #  })
  #}

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

    assert_true(!is.null(.input_map))
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

#' @include utils.R
data_descriptor_union = function(dd1, dd2) {
  # Otherwise it is ugly to do the caching of the data loading
  # and this is not really a strong restriction
  assert_true(dd1$.dataset_hash == dd2$.dataset_hash)
  g1 = dd1$graph
  g2 = dd2$graph

  input_map = unique(c(
    set_names(dd1$.input_map, g1$input$name),
    set_names(dd2$.input_map, g2$input$name)
  ))

  graph = merge_graphs(g1, g2) # shallow clone, g1 and g2 graphs (not pipeops) are unmodified

  DataDescriptor(
    dataset = dd1$dataset,
    dataset_shapes = dd1$dataset_shapes,
    graph = graph,
    .input_map = input_map,
    .pointer = dd1$.pointer,
    .pointer_shape = dd1$.pointer_shape,
    .pointer_shape_predict = dd1$.pointer_shape_predict,
    clone_graph = FALSE
  )
}

#' @export
#' @include utils.R
print.DataDescriptor = function(x, ...) {
  catn(sprintf("<DataDescriptor: %d ops>", length(x$graph$pipeops)))
  catn(sprintf("* dataset_shapes: %s", shape_to_str(x$dataset_shapes)))
  catn(sprintf("* .input_map: (%s) -> Graph", paste0(x$.input_map, collapse = ", ")))
  catn(sprintf("* .pointer: %s", paste0(x$.pointer, collapse = ".")))
  catn(str_indent("* .shape(train):",
    if (is.null(x$.pointer_shape)) "<unknown>" else shape_to_str(list(x$.pointer_shape))))
  catn(str_indent("* .shape(predict):",
    if (is.null(x$.pointer_shape_predict)) "<unknown>" else shape_to_str(list(x$.pointer_shape_predict))))
}

# TODO: printer

set_data_descriptor_hash = function(data_descriptor) {
  # avoid partial argument matching
  data_descriptor$.hash = calculate_hash(
    data_descriptor[[".dataset_hash"]],
    data_descriptor[["graph"]][["hash"]],
    data_descriptor[[".input_map"]]
  )
  return(data_descriptor)
}
