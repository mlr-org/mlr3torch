#' @title Materialize Lazy Tensor Columns
#' @description
#' This will materialize a [`lazy_tensor()`] or a `data.frame()` / `list()` containing -- among other things --
#' [`lazy_tensor()`] columns.
#' I.e. the data described in the underlying [`DataDescriptors`] is loaded for the indices in the [`lazy_tensor()`],
#' is preprocessed and then put unto the specified device.
#' Because not all elements in a lazy tensor must have the same shape, a list of tensors is returned by default.
#' If all elements have the same shape, these tensors can also be rbinded into a single tensor (parameter `rbind`).
#'
#' @details
#' Materializing a lazy tensor consists of:
#' 1. Loading the data from the internal dataset of the [`DataDescriptor`].
#' 2. Processing these batches in the preprocessing [`Graph`]s.
#' 3. Returning the result of the [`PipeOp`] pointed to by the [`DataDescriptor`] (`.pointer`).
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
#' @return (`list()` of [`lazy_tensor`]s or a [`lazy_tensor`])
#' @export
#' @examples
#' lt1 = as_lazy_tensor(torch_randn(10, 3))
#' materialize(lt1, rbind = TRUE)
#' materialize(lt1, rbind = FALSE)
#' lt2 = as_lazy_tensor(torch_randn(10, 4))
#' d = data.frame(lt1 = lt1, lt2 = lt2)
#' materialize(d, rbind = TRUE)
#' materialize(d, rbind = FALSE)
materialize = function(x, device = "cpu", rbind = FALSE, ...) {
  assert_choice(device, mlr_reflections$torch$devices)
  assert_flag(rbind)
  UseMethod("materialize")
}

#' @param cache (`character(1)` or `environment()` or `NULL`)\cr
#'   Optional cache for (intermediate) materialization results.
#'   Per default, caching will be enabled when the same dataset / graph is used for more than one lazy tensor column.
#' @export
materialize.list = function(x, device = "cpu", rbind = FALSE, cache = "auto") { # nolint
  x_lt = x[map_lgl(x, is_lazy_tensor)]
  assert(check_choice(cache, "auto"), check_environment(cache, null.ok = TRUE))

  if (identical(cache, "auto")) {
    data_hashes = map_chr(x_lt, function(x) x$.dataset_hash)
    hashes = map_chr(x_lt, function(x) x$.hash)
    cache = if (uniqueN(data_hashes) > 1L || uniqueN(hashes) > 1L) {
      new.env()
    }
  }

  # if we are materializing more than one lazy tensor, we must specify what we want to keep from the graph
  # BEFORE calling materialize_internal because materialize_internal does not know about the other columns
  keep_results_prev = list()
  keep_results_prev = map(x_lt, function(x) {
    graph = x$graph
    keep_results = graph$keep_results
    graph$keep_results = character(0)
    return(keep_results)
  })
  on.exit({
    walk(seq_along(x_lt), function(i) {
      graph = x_lt[[i]]$graph
      graph$keep_results = keep_results_prev[[i]]
    })
  }, add = TRUE)

  walk(x_lt, function(col) {
    graph = col$graph
    graph$keep_results = union(graph$keep_results, col$.pointer[1L])
  })

  map(x, function(col) {
    if (is_lazy_tensor(col)) {
      browser()
      materialize_internal(col, device = device, cache = cache, set_keep_results = FALSE, rbind = rbind)
    } else {
      col
    }
  })
}




#' @method materialize data.frame
#' @export
materialize.data.frame = function(x, device = "cpu", rbind = FALSE, cache = "auto") { # nolint
  materialize(as.list(x), device = device, rbind = rbind, cache = cache)
}


#' @export
materialize.lazy_tensor = function(x, device = "cpu", rbind = FALSE) { # nolint
  materialize_internal(x = x, device = device, cache = NULL, set_keep_results = TRUE, rbind = rbind)
}

#' @title Materialize a Lazy Tensor
#' @description
#' Convert a [`lazy_tensor()`] to a [`torch_tensor()`].
#'
#' @details
#' Materializing a lazy tensor consists of:
#' 1. Loading the data from the internal dataset of the [`DataDescriptor`].
#' 2. Processing these batches in the preprocessing [`Graph`]s.
#' 3. Returning the result of the [`PipeOp`] pointed to by the [`DataDescriptor`] (`.pointer`).
#'
#' When materializing multiple [`lazy_tensor`] columns, caching can be useful because:
#' a) Output(s) from the dataset might be input to multiple graphs.
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
#' @param set_keep_results (`logical(1)`)\cr
#'   In some cases, the `.pointer` of a [`DataDescriptor`] might point to a non-terminal node in which case the
#'   this result is not part of the output of the [`Graph`].
#'   Therefore we have to include this as part of the `keep_results` field of the [`Graph`].
#'   When caching is done, this should be set to `FALSE` as otherwise data will be discarded that might be relevant
#'   for materializing other lazy tensor columns.
#' @param rbind (`logical(1)`)\cr
#'   Whtether to rbind the resulting tensors (`TRUE`) or return them as a list of tensors (`FALSE`).
#' @return [`lazy_tensor()`]
#' @keywords internal
materialize_internal = function(x, device = "cpu", cache = NULL, set_keep_results = is.null(cache), rbind) {
  if (!length(x)) {
    stopf("Cannot materialize lazy tensor of length 0.")
  }
  do_caching = !is.null(cache)
  ids = map_int(vec_data(x), 1)

  data_descriptor = x$data_descriptor
  ds = data_descriptor$dataset
  graph = data_descriptor$graph
  varying_shapes = some(data_descriptor$dataset_shapes, function(shape) all(is.na(shape)))

  if (set_keep_results) {
    prev_results = graph$keep_results
    on.exit({graph$keep_results = prev_results}, add = TRUE) # nolint
    graph$keep_results = data_descriptor$.pointer[1L]
  }

  if (do_caching) {
    output_hash = calculate_hash(ids, data_descriptor$.hash)
    output_hit = exists(output_hash, cache, inherits = FALSE)

    if (output_hit) {
      return(cache[[output_hash]][[data_descriptor$.pointer]])
    }

    input_hash = calculate_hash(data_descriptor$.dataset_hash, ids)
    input_hit = exists(input_hash, cache, inherits = FALSE)

    if (input_hit) {
      input = cache[[input_hash]]
      input_hit = TRUE
    }
  }

  if (!do_caching || !input_hit) {
    input = if (is.null(ds$.getbatch)) { # .getindex is never NULL but a function that errs if it was not defined
      x = map(ids, function(id) map(ds$.getitem(id), function(x) x$unsqueeze(1)))
      if (varying_shapes || !rbind) {
        x
      } else {
        map(transpose_list(x), function(x) torch_cat(x, dim = 1L))
      }
    } else {
      if (rbind) {
        ds$.getbatch(ids)
      } else {
        map(ids, function(id) ds$.getbatch(id))
      }
    }
  }

  if (do_caching && !input_hit) {
    cache[[input_hash]] = input
  }

  # input is the output of a dataset so it can contain more than what we need for the graph,
  # also we need to set the correct names.
  # This is done after retrieving the element from the cache / before saving the element to the cache because
  # this can change

  input = if (rbind && !varying_shapes) {
    set_names(input[data_descriptor$.input_map], data_descriptor$.graph_input)
  } else {
    map(input, function(x) {
      set_names(x[data_descriptor$.input_map], data_descriptor$.graph_input)
    })
  }

  output = if (rbind && !varying_shapes) {
    # tensor --graph--> tensor
    graph$train(input, single_input = FALSE)
    out = map(data_descriptor$graph$keep_results, function(id) graph$pipeops[[id]]$.result)
    set_names(out, data_descriptor$graph$keep_results)
  } else {
    # list --graph--> (list or tensor)
    out = map(input, function(x) {
      graph$train(x, single_input = FALSE)
      out = map(data_descriptor$graph$keep_results, function(id) graph$pipeops[[id]]$.result)
      set_names(out, data_descriptor$graph$keep_results)
    })
    if (rbind) {
      # here, is a list with hierarchy: [id = [po_id = [ch_nm = ]]]
      # We want to obtain a list [po_id = [ch_nm = [...]]] where the [...] is the rbind over all ids
      out = map(names(out[[1L]]), function(po_id) {
        map(names(out[[1]][[po_id]]), function(ch_nm) {
          torch_cat(map(seq_along(out), function(i) out[[i]][[po_id]][[ch_nm]]), dim = 1L)
        })
      })
    }
    out
  }

  if (do_caching) {
    cache[[output_hash]] = output
  }

  # discard the results
  walk(data_descriptor$graph$pipeops[data_descriptor$graph$keep_results], function(x) x$.result = NULL)

  if (rbind) {
    res = output[[data_descriptor$.pointer[1L]]][[data_descriptor$.pointer[2L]]]$to(device = device)
  } else {
    res = map(output, function(o) o[[data_descriptor$.pointer[1L]]][[data_descriptor$.pointer[2L]]]$to(device = device))
  }

  return(res)
}
