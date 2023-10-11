#' @title Materialize Lazy Tensor Columns
#' @description
#' This will materialize all columns from a `data.frame` containing [`lazy_tensor()`] columns.
#' @details
#' Caching is applied to ensure that each batch from the datasets wrapped by the lazy tensors is only loaded once.
#' It is also ensured that each graph is only applied once to a given input batch, i.e. different columns
#' can share the same graph, dataset and indices but point to different outputs.
#' @export
materialize = function(x, device = "cpu", ...) {
  UseMethod("materialize")
}

#' @export
materialize.list = function(x, device = "cpu") {
  input_cache = new.env() # nolint
  output_cache = new.env() # nolint

  keep_results_prev = list()
  walk(x, function(col) {
    keep+
    if (is_lazy_tensor(col)) {
      keep_results_prev[[graph$hash]] = graph$keep_results
      attr(col, "data_descriptor")$graph$keep_results = character()

    }
  })
  # clean up the keep_results after
  on.exit({walk(x, function(col) {
    if (is_lazy_tensor(x)) {
      graph = attr(x, "data_descriptor")$graph
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
    if (!is_lazy_tensor(col)) {
      return(col)
    }
    data_descriptor = attr(col, "data_descriptor")
    dataset_hash = data_descriptor$.dataset_hash

    ds = data_descriptor$dataset

    ids = vec_data(col)

    input_hash = calculate_hash(dataset_hash, ids)

    if (!exists(input_hash, input_cache, inherits = FALSE)) {
      input = if (is.null(ds$.getbatch)) { # .getindex is never NULL but a function that errs if it was not defined
        torch_cat(map(ids, function(id) ds$.getitem(id)), dim = 1L)
      } else {
        ds$.getbatch(ids)
      }

      input_cache[[input_hash]] = input
    } else {
      input = input_cache[[input_hash]]
    }

    input = set_names(input[data_descriptor$.input_map], data_descriptor$graph$.graph_input)

    output_hash = calculate_hash(ids, data_descriptor$.hash)

    if (!exists(output_hash, output_cache, inherits = FALSE)) {
      data_descriptor$graph$train(input, single_input = FALSE)
      output = map(data_descriptor$graph$pipeops[data_descriptor$graph$keep_results], ".result")
      output_cache[[output_hash]] = output
    } else {
      output = output_cache[[output_hash]]
    }
    # FIXME: This will fail when the .pointer is not an output of the graph
    # https://github.com/mlr-org/mlr3torch/issues/138

    output[[data_descriptor$.pointer[1L]]][[data_descriptor$.pointer[2L]]]$to(device = device)
  })
}

#' @export
materialize.lazy_tensor = function(x, device = "cpu", cache = NULL) {
  do_caching = !is.null(cache)
  ids = vec_data(x)

  data_descriptor = attr(x, "data_descriptor")
  ds = data_descriptor$dataset
  graph = data_descriptor$graph
  prev_results = graph$keep_results
  on.exit({graph$keep_results = prev_results}, add = TRUE)
  if (test_flag(prev_results)) {
    graph$keep_results = data_descriptor$.pointer[1L]
  } else {
    graph$keep_results = union(prev_results, data_descriptor$.pointer[1L])
  }

  if (do_caching) {
    output_hash = calculate_hash(ids, data_descriptor$.hash)
    output_hit = exists(output_hash, cache$output, inherits = FALSE)

    if (output_hit) {
      return(cache$output[[output_hash]][[data_descriptor$.pointer]])
    }

    input_hash = calculate_hash(dataset_hash, ids)
    input_hit = exists(input_hash, cache$input, inherits = FALSE)

    if (input_hit) {
      input = cache$input[[input_hash]]
      input_hit = TRUE
    }
  }

  if (!do_caching || !input_hit) {
    input = if (is.null(ds$.getbatch)) { # .getindex is never NULL but a function that errs if it was not defined
      tmp = transpose_list(map(ids, function(id) map(ds$.getitem(id), function(x) x$unsqueeze(1))))
      map(tmp, function(x) torch_cat(x, dim = 1L))
    } else {
      ds$.getbatch(ids)
    }
  }

  if (do_caching && !input_hit) {
    cache$input[[input_hash]] = input
  }

  # input is the output of a dataset so it can contain more than what we need for the graph,
  # also we need to set the correct names.
  # This is done after retrieving the element from the cache / before saving the element to the cache because
  # this can change

  browser()
  input = set_names(input[data_descriptor$.input_map], data_descriptor$.graph_input)

  graph$train(input, single_input = FALSE)
  output = map(data_descriptor$graph$keep_results, function(id) graph$pipeops[[id]]$.result)
  output = set_names(output, data_descriptor$graph$keep_results)

  if (do_caching) {
    cache$output[[output_hash]] = output
  }

  # discard the results
  walk(data_descriptor$graph$pipeops[data_descriptor$graph$keep_results], function(x) x$.result = NULL)

  output = output[[data_descriptor$.pointer[1L]]][[data_descriptor$.pointer[2L]]]$to(device = device)
}


#' @method materialize data.frame
#' @export
materialize.data.frame = materialize.list


batchgetter_lazy_tensor = function(data, device, cache = NULL) {
  if (is.null(cache)) {
    materialize(data[[1L]])[[1L]]


  }
}
