#' @title Materialize Lazy Tensor Columns
#' @description
#' This will materialize a [`lazy_tensor()`] or a `data.frame()` / `list()` containing -- among other things --
#' [`lazy_tensor()`] columns.
#' I.e. the data described in the underlying [`DataDescriptor`]s is loaded for the indices in the [`lazy_tensor()`],
#' is preprocessed and then put unto the specified device.
#' Because not all elements in a lazy tensor must have the same shape, a list of tensors is returned by default.
#' If all elements have the same shape, these tensors can also be rbinded into a single tensor (parameter `rbind`).
#'
#' @details
#' Materializing a lazy tensor consists of:
#' 1. Loading the data from the internal dataset of the [`DataDescriptor`].
#' 2. Processing these batches in the preprocessing [`Graph`][mlr3pipelines::Graph]s.
#' 3. Returning the result of the [`PipeOp`][mlr3pipelines::PipeOp] pointed to by the [`DataDescriptor`] (`pointer`).
#'
#' With multiple [`lazy_tensor`] columns we can benefit from caching because:
#' a) Output(s) from the dataset might be input to multiple graphs.
#' b) Different lazy tensors might be outputs from the same graph.
#'
#' For this reason it is possible to provide a cache environment.
#' The hash key for a) is the hash of the indices and the dataset.
#' The hash key for b) is the hash of the indices, dataset and preprocessing graph.
#'
#' @param x (any)\cr
#'   The object to materialize.
#'   Either a [`lazy_tensor`] or a `list()` / `data.frame()` containing [`lazy_tensor`] columns.
#' @param rbind (`logical(1)`)\cr
#'   Whether to rbind the lazy tensor columns (`TRUE`) or return them as a list of tensors (`FALSE`).
#'   In the second case, there is no batch dimension.
#' @return (`list()` of [`lazy_tensor`]s or a [`lazy_tensor`])
#' @param device (`character(1)`)\cr
#'   The torch device.
#' @param ... (any)\cr
#'   Additional arguments.
#' @export
#' @examplesIf torch::torch_is_installed()
#' lt1 = as_lazy_tensor(torch_randn(10, 3))
#' materialize(lt1, rbind = TRUE)
#' materialize(lt1, rbind = FALSE)
#' lt2 = as_lazy_tensor(torch_randn(10, 4))
#' d = data.table::data.table(lt1 = lt1, lt2 = lt2)
#' materialize(d, rbind = TRUE)
#' materialize(d, rbind = FALSE)
materialize = function(x, device = "cpu", rbind = FALSE, ...) {
  assert_choice(device, mlr_reflections$torch$devices)
  assert_flag(rbind)
  UseMethod("materialize")
}

#' @rdname materialize
#' @param cache (`character(1)` or `environment()` or `NULL`)\cr
#'   Optional cache for (intermediate) materialization results.
#'   Per default, caching will be enabled when the same dataset or data descriptor (with different output pointer)
#'   is used for more than one lazy tensor column.
#' @export
materialize.list = function(x, device = "cpu", rbind = FALSE, cache = "auto", ...) { # nolint
  x_lt = x[map_lgl(x, is_lazy_tensor)]
  assert(check_choice(cache, "auto"), check_environment(cache, null.ok = TRUE))

  if (identical(cache, "auto")) {
    cache = if (auto_cache_lazy_tensors(x_lt)) new.env()
  }

  map(x, function(col) {
    if (is_lazy_tensor(col)) {
      materialize_internal(col, device = device, cache = cache, rbind = rbind)
    } else {
      col
    }
  })
}




#' @method materialize data.frame
#' @export
materialize.data.frame = function(x, device = "cpu", rbind = FALSE, cache = "auto", ...) { # nolint
  materialize(as.list(x), device = device, rbind = rbind, cache = cache)
}


#' @export
materialize.lazy_tensor = function(x, device = "cpu", rbind = FALSE, ...) { # nolint
  materialize_internal(x = x, device = device, cache = NULL, rbind = rbind)
}

get_input = function(ds, ids, varying_shapes, rbind) {
  if (is.null(ds$.getbatch)) { # .getindex is never NULL but a function that errs if it was not defined
    x = map(ids, function(id) map(ds$.getitem(id), function(x) x$unsqueeze(1)))
    if (varying_shapes) {
      x
    } else {
      map(transpose_list(x), function(x) torch_cat(x, dim = 1L))
    }
  } else {
    ds$.getbatch(ids)
  }
}

get_output = function(input, graph, varying_shapes, rbind, device) {
  output = if (varying_shapes) {
    # list --graph--> (list or tensor)
    transpose_list(map(input, function(x) graph$train(x, single_input = FALSE)))
  } else {
    # tensor --graph--> tensor
    graph$train(input, single_input = FALSE)
  }

  # now we get it in the right output format and convert it to the requested device
  output = if (rbind) {
    if (varying_shapes) { # need to convert from list of tensors to tensor
      output = map(output, function(tensors) torch_cat(tensors, dim = 1L))
    }
    map(output, function(x) x$to(device = device))
  } else {
    if (!varying_shapes) { # need to convert from tensor to list of tensors
      output = map(output, function(x) torch_split(x, split_size = 1L, dim = 1L))
    }
    map(output, function(out) map(out, function(o) o$squeeze(1)$to(device = device)))
  }

  return(output)
}

#' @title Materialize a Lazy Tensor
#' @description
#' Convert a [`lazy_tensor`] to a [`torch_tensor`][torch::torch_tensor].
#'
#' @details
#' Materializing a lazy tensor consists of:
#' 1. Loading the data from the internal dataset of the [`DataDescriptor`].
#' 2. Processing these batches in the preprocessing [`Graph`][mlr3pipelines::Graph]s.
#' 3. Returning the result of the [`PipeOp`][mlr3pipelines::PipeOp] pointed to by the [`DataDescriptor`] (`pointer`).
#'
#' When materializing multiple [`lazy_tensor`] columns, caching can be useful because:
#' a) Output(s) from the dataset might be input to multiple graphs.
#'    (in task_dataset this is shoudl rarely be the case because because we try to merge them).
#' b) Different lazy tensors might be outputs from the same graph.
#'
#' For this reason it is possible to provide a cache environment.
#' The hash key for a) is the hash of the indices and the dataset.
#' The hash key for b) is the hash of the indices dataset and preprocessing graph.
#'
#' @param x ([`lazy_tensor()`])\cr
#'   The lazy tensor to materialize.
#' @param device (`character(1L)`)\cr
#'   The device to put the materialized tensor on (after running the preprocessing graph).
#' @param cache (`NULL` or `environment()`)\cr
#'   Whether to cache the (intermediate) results of the materialization.
#'   This can make data loading faster when multiple `lazy_tensor`s reference the same dataset or graph.
#' @param rbind (`logical(1)`)\cr
#'   Whtether to rbind the resulting tensors (`TRUE`) or return them as a list of tensors (`FALSE`).
#' @return [`lazy_tensor()`]
#' @keywords interna
materialize_internal = function(x, device = "cpu", cache = NULL, rbind) {
  if (!length(x)) {
    stopf("Cannot materialize lazy tensor of length 0.")
  }
  do_caching = !is.null(cache)
  ids = map_int(x, 1)

  data_descriptor = dd(x)
  ds = data_descriptor$dataset
  graph = data_descriptor$graph
  varying_shapes = some(data_descriptor$dataset_shapes, is.null)

  pointer_name = paste0(data_descriptor$pointer, collapse = ".")
  if (do_caching) {
    output_hash = calculate_hash(ids, data_descriptor$hash)
    output_hit = exists(output_hash, cache, inherits = FALSE)

    if (output_hit) {
      return(cache[[output_hash]][[pointer_name]])
    }
    input_hash = calculate_hash(data_descriptor$dataset_hash, ids)

    input_hit = exists(input_hash, cache, inherits = FALSE)

    if (input_hit) {
      input = cache[[input_hash]]
    }
  }

  if (!do_caching || !input_hit) {
    input = get_input(ds, ids, varying_shapes, rbind)
  }

  if (do_caching && !input_hit) {
    cache[[input_hash]] = input
  }

  # input is the output of a dataset so it can contain more than what we need for the graph,
  # also we need to set the correct names.
  # This is done after retrieving the element from the cache / before saving the element to the cache because
  # this can change depending on the preprocessing graph
  input = if (varying_shapes) {
    map(input, function(x) {
      set_names(x[data_descriptor$input_map], data_descriptor$graph_input)
    })
  } else {
    set_names(input[data_descriptor$input_map], data_descriptor$graph_input)
  }

  output = get_output(input, graph, varying_shapes, rbind, device)

  if (do_caching) {
    cache[[output_hash]] = output
  }

  output[[pointer_name]]
}
