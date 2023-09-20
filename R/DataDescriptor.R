#' @title Data Descriptor
#'
#' @description
#' A data descriptor is a rather internal structure used in the [`lazy_tensor`] data type.
#' In essence it is an annotated [`torch::dataset`] and a preprocessing graph (consisting mosty of [`PipeOpModule`]
#' operators). The additional meta data (e.g. shapes) allows to preprocess [`lazy_tensors`] in an
#' [`mlr3pipelines::Graph`] just like any (non-lazy) data types.
#'
#' @param dataset ([`torch::dataset`])\cr
#'   The torch dataset.
#' @param dataset_shapes (named `list()` of `integer()`s)\cr
#'   The shapes of the output.
#'   Names are the elements of the list returned by the dataset.
#'   First dimension must be `NA`.
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
#'   Shape of the output indicated by `.pointer`. Note that this is **without** the batch dimension as opposed
#'   to the [`ModelDescriptor`]. The reason is that the .pointer_shape refers to exactly one element and hence
#'   has no batch dimension.
#' @param clone_graph (`logical(1)`)\cr
#'   Whether to clone the preprocessing graph.
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
  .pointer_shape = NULL, clone_graph = TRUE) {
  assert_class(dataset, "dataset")
  assert_shapes(dataset_shapes)

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
    graph = as_graph(graph)
    if (clone_graph) {
      graph = graph$clone(deep = TRUE)
    }
    assert_true(length(graph$pipeops) >= 1L)

    if (any(is.null(.input_map), is.null(.pointer), is.null(.pointer_shape))) {
      stopf("When passing a graph you need to specify .input_map, .pointer and .pointer_shape.")
    }

    assert_choice(.pointer[[1]], names(graph$pipeops))
    assert_choice(.pointer[[2]], graph$pipeops[[.pointer[[1]]]]$output$name)
    assert_subset(paste0(.pointer, collapse = "."), graph$output$name)
    assert_integerish(.pointer_shape, min.len = 1L)

    assert_subset(.input_map, names(dataset_shapes))
    assert_true(length(.input_map) == length(graph$input$name))
  }

  # We get a warning that package:mlr3torch may not be available when loading (?)
  dataset_hash = suppressWarnings(calculate_hash(dataset, dataset_shapes))
  obj = structure(
    list(
      dataset = dataset,
      graph = graph,
      dataset_shapes = dataset_shapes,
      .input_map = .input_map,
      .pointer = .pointer,
      .pointer_shape = .pointer_shape,
      .dataset_hash = dataset_hash,
      .hash = NULL # is set below
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
