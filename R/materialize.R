#' @title Materialize Lazy Tensor Columns
#' @description
#' This will materialize a [`lazy_tensor()`] or a [`data.frame()`] / [`list()`] containing -- among other things --
#' [`lazy_tensor()`] columns.
#' I.e. the data described in the underlying [`DataDescriptors`] is loaded for the indices in the [`lazy_tensor()`]
#' and is afterwards put on the specified device.
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
#' The hash key for b) is the hash of the indices dataset and preprocessing graph.
#' @param x (any)\cr
#'   The object to materialize.
#'   Either a [`lazy_tensor`] or a `list()` / `data.frame()` containing [`lazy_tensor`] columns.
#' @param rbind (`logical(1)`)\cr
#'   Whether to rbind the lazy tensor columns (`TRUE`) or return them as a list of tensors (`FALSE`).
#' @return ([`torch_tensor()`] or [`list()`])
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

#' @export
materialize.list = function(x, device = "cpu", rbind = FALSE) { # nolint
  # FIXME: smarter cache detecting
  cache = if (sum(map_lgl(x, is_lazy_tensor)) > 1L) {
    new.env()
  }

  keep_results_prev = list()
  walk(x, function(col) {
    if (is_lazy_tensor(col)) {
      graph = attr(col, "data_descriptor")$graph
      keep_results_prev[[graph$hash]] = graph$keep_results
      graph$keep_results = character()
    }
  })
  # clean up the keep_results after
  on.exit({walk(x, function(col) { # nolint
    if (is_lazy_tensor(x)) {
      graph = x$graph
      graph$keep_results = keep_results_prev[[graph$hash]]
    }
  })}, add = TRUE)

  walk(x, function(col) {
    if (is_lazy_tensor(col)) {
      dd = attr(col, "data_descriptor")
      dd$graph$keep_results = union(dd$graph$keep_results, dd$.pointer[1L])
    }
  })

  # TODO: No hashing when there is only one column
  map(x, function(col) {
    if (is_lazy_tensor(col)) {
      materialize_internal(col, device = device, cache = cache, set_keep_results = FALSE, rbind = rbind)
    } else {
      col
    }
  })
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
  ids = vec_data(x)

  data_descriptor = attr(x, "data_descriptor")
  ds = data_descriptor$dataset
  graph = data_descriptor$graph

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
      if (rbind) {
        map(transpose_list(x), function(x) torch_cat(x, dim = 1L))
      } else {
        x
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

  input = if (rbind) {
    set_names(input[data_descriptor$.input_map], data_descriptor$.graph_input)
  } else {
    map(input, function(x) {
      set_names(x[data_descriptor$.input_map], data_descriptor$.graph_input)
    })
  }

  output = if (rbind) {
    # in this case, input is a tensor
    graph$train(input, single_input = FALSE)
    out = map(data_descriptor$graph$keep_results, function(id) graph$pipeops[[id]]$.result)
    set_names(out, data_descriptor$graph$keep_results)
  } else {
    # in this case, input is a list of tensors
    map(input, function(x) {
      graph$train(x, single_input = FALSE)
      out = map(data_descriptor$graph$keep_results, function(id) graph$pipeops[[id]]$.result)
      set_names(out, data_descriptor$graph$keep_results)
    })
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


#' @method materialize data.frame
#' @export
materialize.data.frame = materialize.list # nolint
