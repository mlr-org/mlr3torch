#' @title Materialize a Lazy Tensor
#' @description
#' Materialize a [`lazy_tensor`] or a [`data.tabe`] with lazy tensor columns.
#' This loads the data and then applies the preprocessing graph and returns its output.
#'
#' @param x (any)\cr
#'   The object to materialize.
#' @param cat (`logical(1)`)\cr
#'   Whether to concatenate the torch tensors along the batch dimension.
#'   If `FALSE` (default) a [`data.table`] is returned, otherwise a `list()`.
#' @return `list()` or `data.table()`.
#' @export
#' @seealso lazy_tensor
#' @examples
#' lt1 = as_lazy_tensor(torch_randn(10, 1))
#' materialize(lt1)
materialize = function(x, ...) {
  UseMethod("materialize")
}

# FIXME: Currently this **could** turn out to be very inefficient but premature optimization ...
# At least we should suppot batchwise operations in the simple case (one DataDescriptor)
# where we first create the batches and then process them with the graph
# Also proprely finish the cached version of this below

#' @export
materialize.lazy_tensor = function(x, cat = FALSE) { # nolint
  tensors = map(x, function(elt) {
    data_descriptor = elt$data_descriptor
    graph = data_descriptor$graph
    ds = data_descriptor$dataset
    input = if (is.null(ds$.getbatch)) { # .getindex is never NULL but a function that errs if it was not defined
      map(ds$.getitem(elt$id), function(x) x$unsqueeze(1))[elt$.input_map]
    } else {
      ds$.getbatch(elt$id)[data_descriptor$.input_map]
    }
    input = set_names(input, data_descriptor$graph$input$name)
    graph$train(input, single_input = FALSE)[[1L]]
  })

  if (!cat) {
    return(tensors)
  }
  return(torch_cat(tensors, dim = 1L))
}

#' @method materialize data.table
#' @export
materialize.data.table = function(x, cat = FALSE) {
  cols = map(x, function(x) {
    if (!is_lazy_tensor(x)) {
      return(x)
    } else {
      materialize(x, cat = cat)
    }
  })

  if (!cat) {
    return(as.data.table(cols))
  }

  map(cols, function(col) {
    if (test_class(col, "torch_tensor")) {
      torch_cat(col, dim = 1L)
    } else {
      col
    }
  })
}

#materialize.data.table = function(x, batchwise) { # nolint
#  # TODO: Here it should be possible to specify meta-information that can be precomputed once such as
#  # whether .getbatch should be used (for that we need to compare a lot of hashes and it is enough to do this once)
#  # --> if lazy_tensor supports .getbatch() we need to add it here as well.
#
#  batch_env = new.env()
#
#  # Because we might load the same batches multiple times
#
#  map_dtc(x, function(column) {
#    if (!is_lazy_tensor(column)) {
#      return(column)
#    }
#    ids = unlist(map(column, function(elt) elt[[1]]$id))
#
#    tnsrs = map(column, function(elt) {
#      ids = map()
#      data_descriptor = elt$data_descriptor
#      hash = data_descriptor$.hash
#      id = elt$id
#      id_chr = as.character(id)
#      if (!exists(hash, batch_env)) {
#        batch_env[[hash]] = new.env()
#        batch_env[[hash]][[id_chr]] = data_descriptor$dataset$.getitem(id)$unsqueeze(1)
#      } else if (is.null(batch_env[[hash]][[id_chr]])) {
#        batch_env[[hash]][[id_chr]] = data_descriptor$dataset$.getitem(id)$unsqueeze(1)
#      }
#      batch_env[[hash]][[id_chr]][[elt$.pointer]]
#    })
#    torch_cat(tnsrs, dim = 1L)
#  })
#}
#
## Just retrieve
#cache_data_default = function(elt, input_cache) {
#  data_descriptor = elt$data_descriptor
#  dataset_hash = data_descriptor$.dataset_hash
#  id = elt$id
#  id_chr = as.character(id)
#
#  if (!exists(dataset_hash, input_cache)) {
#    input_cache[[dataset_hash]] = new.env()
#    if (!is.null(data_descriptor$dataset$.getbatch)) {
#      input_cache[[dataset_hash]][[id_chr]] = data_descriptor$dataset$.getbatch(id)
#    } else {
#      input_cache[[dataset_hash]][[id_chr]] = data_descriptor$dataset$.getitem(id)$unsqueeze(1)
#    }
#  } else if (is.null(input_cache[[dataset_hash]][[id_chr]])) {
#    if (!is.null(data_descriptor$dataset$.getbatch)) {
#      input_cache[[dataset_hash]][[id_chr]] = data_descriptor$dataset$.getbatch(id)
#    } else {
#      input_cache[[dataset_hash]][[id_chr]] = data_descriptor$dataset$.getitem(id)$unsqueeze(1)
#    }
#  }
#}
#
## Assumption: Each column in x has exactly one batch
#materialize_data_table_batchwise = function(x) {
#  # Use retrieve_batch with a cache
#  # A single graph (because all )
#
#}
#
#materialize_data_table_default = function(x) {
#  # First we retrieve all elements that we need here.
#  input_cache = new.env()
#
#  # we first obtain all the batches from the datasets and store them in a hash table with the structure:
#  # table:
#  #   * dataset_hash1 : [id1, id2, ...]
#  #   * dataset_hash2 : [id1, id2, ...]
#  walk(x, function(column) {
#    if (!is_lazy_tensor(column)) {
#      return(NULL)
#    }
#    walk(column, function(elt) {
#      cache_data_default(elt, input_cache)
#    })
#  })
#  browser()
#
#  # Now we have the inputs and then process all the graphs using the inputs.
#  output_cache = new.env()
#
#  map_dtc(x, function(column) {
#    if (!test_class(column, "lazy_tensor")) {
#      return(column)
#    }
#    tnsrs = map(column, function(elt) {
#      # the graph hash consists of the dataset, graph and input map
#      # it excludes the .pointer as differet elements might point to different outputs from the same graph / data combo
#      preprocess_and_cache_default(elt, input_cache = input_cache, output_cache = output_cache)
#    })
#
#    torch_cat(tnsrs, dim = 1L)
#  })
#}
#
#preprocess_and_cache_default = function(elt, input_cache, output_cache) {
#  # retrieve input to the graph
#  data_descriptor = elt$data_descriptor
#  graph = data_descriptor$graph
#
#  # FIXME: this currently ignores the .pointer and always assumes that there is only one graph output
#  # To fix this we need to modify the graph in materailize_data_table to add NOPs for all the .pointers.
#  # An easier way would be to allow specifying which outputs to keep when calling graph$train()
#  # However this means that the hash used to identify this should not include the .pointer
#  # as otherwise no time is safed
#  assert_true(isTRUE(all.equal(paste0(data_descriptor$.pointer, collapse = "."), graph$output$name)))
#  id = elt$id
#  id_chr = as.character(id)
#  x = set_names(input_cache[[data_descriptor$.dataset_hash]][[id_chr]][data_descriptor$.input_map],
#    graph$input$name)
#
#
#  if (!exists(data_descriptor$.hash, output_cache)) {
#    output_cache[[data_descriptor$.hash]] = new.env()
#    output_cache[[data_descriptor$.hash]][[id_chr]] = graph$train(x, single_input = FALSE)[[1L]]
#  } else if (is.null(output_cache[[data_descriptor$.hash]][[id_chr]])) {
#    output_cache[[data_descriptor$.hash]][[id_chr]] = graph$train(x, single_input = FALSE)[[1L]]
#  }
#
#  output_cache[[data_descriptor$.hash]][[id_chr]][[data_descriptor$.pointer]]
#}
#
