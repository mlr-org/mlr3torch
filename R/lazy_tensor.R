#' Creates a lazy tensor
#'
#' @description
#' Creates an object of class `"lazy_tensor"`.
#' This allows to represent data that is stored in
#'
#' @param x ([`DataDescriptor`])\cr
#'   The data descriptor.
#' @param id (`vector()`)\cr
#'   An array with unique identifiers.
#' @param column (`character()`)\cr
#'   The column(s) of the [`TorchDataset`] that this vector represents.
#'   Per default the column of the dataset descriptor is chosen if there is only one.
#'
#' @return [`lazy_tensor`]
#' @export
lazy_tensor = function(data_descriptor, ids = NULL) {
  assert_class(data_descriptor, "DataDescriptor")
  if (is.null(ids)) {
    ids = seq_along(dataset_descriptor$dataset)
  } else {
    assert_vector(id, len = length(data_descriptor$dataset))
  }

  obj = map(ids, function(id) {
    list(id = id, dataset_descriptor = dataset_descriptor)
  })

  obj = structure(
    obj,
    class = c("lazy_tensor", "list")
  )
}

#' @export
DataDescriptor = function(dataset, dataset_shapes, graph = NULL, input_map = NULL, output = NULL, .pointer = NULL, .pointer_shape = NULL) {
  # We don't include the id in this data structure because this means 99% of the time all objects
  # in a column are identical
  assert_class(dataset, "TorchDataset")
  if (is.null(graph)) {
    graph = as_graph(po("nop"))
  } else {
    assert_graph(graph)
  }
  assert_list(input_map, null.ok = TRUE)

  if (!is.null(.pointer)) {
    assert_integerish(.pointer_shape)
    assert_choice(.pointer[[1]], names(graph$pipeops))
    assert_choice(.pointer[[2]], graph$pipeops[[.pointer[[1]]]]$output$name)
  }

  hash = calculate_hash(
    dataset,
    data.table::address(graph),
    input_map,
    output,
    .pointer,
    .pointer_shape,
    dataset_shapes
  )


  structure(
    list(
      dataset = dataset,
      graph = graph,
      input_map = input_map,
      output = output,
      .pointer = .pointer,
      .pointer_shape = .pointer_shape,
      # dataset information
      .dataset_shapes = dataset_shapes,
      .hash = hash
    ), class = "DataDescriptor")
}

`%>>%.DataDescriptor`

data_descriptor_union = function(lt1, lt2) {
  if (!identical(lt1$dataset1, lt2$dataset2)) {
    stopf("Can currently only combine data descriptors with the same dataset.")
  }

  graph = merge_graphs(lt1$graph, lt2$graph)


  DatasetDescripor(
    dataset = lt1$dataset1,
    dataset_shapes = lt1$dataset_shapes,
    graph = graph,
    input_map = merge_assert_unique(lt1$input_map, lt2$input_map, "Graphs have compatible input maps.")
  )

}


# TODO: printer for lazy tensor

`c.lazy_tensor` = function(...) {
  if (length(unique(map(list(...), "shape"))) > 1L) {
    stopf("Can only concatenate lazy tensors with the same shape.")
  }
  c(...)
}
