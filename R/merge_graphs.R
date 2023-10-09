#' @title Merge Lazy Tensor Graphs
#' @description
#' Merge lazy tensor graphs for efficiency.
#' @details
#' When preprocessing graphs of multiple lazy tensors have significant overlap (note that we always clone them
#' so they work just like standard preprocessing pipeops) we merge these graphs again to avoid unnecessary
#' computation.
#' @include ModelDescriptor.R
merge_lazy_tensor_graphs = function(dt) {
  browser()
  dt_lazy = dt[, map_lgl(dt, is_lazy_tensor), with = FALSE]

  if (ncol(dt_lazy) <= 1L) {
    return(dt)
  }

  for (col in dt_lazy) {
    n = uniqueN(map_chr(col, function(x) x$data_descriptor$.hash))

    # This is not necessary but we keep it simple for now and almost all cases should have one DataDescriptor per column
    if (n > 1L) {
      lg$warn("Cannot merge lazy tensor graphs, resulting preprocessing might be less efficient.")
      return(dt)
    }
  }

  # TODO:
  # * [ ] What does $clone(deep = TRUE) actually do with PipeOpModule ? Does it modify the hash,
  # then we need to change merge_graphs()
  # * [ ] Check that the mapping of inputs --> input pipeops agree between the pipeops
  # * [ ] Ensure that there is only ONE resulting final graph
  # * [ ] Merge the input_maps
  # * [ ] Add nop outputs for the non-terminal pipeops that have a .pointer pointing to them

  graphs = map(dt_lazy, function(col) {
    col[[1L]]$data_descriptor$graph
  })

  Reduce(merge_graphs, graphs)
}

#' note that this mo
merge_graphs = function(g1, g2, in_place = FALSE) {
  if (in_place) {
    graph = g1
  } else {
    graph = g1$clone(deep = TRUE)
  }
  # if graphs are identical, we don't need to worry about copying stuff
  if (!identical(g1, g2)) {
    # PipeOps that have the same ID that occur in both graphs must be identical.
    common_names = intersect(names(graph$pipeops), names(g2$pipeops))
    if (!identical(map(graph$pipeops[common_names], "hash"), map(g2$pipeops[common_names], "hash"))) {
      not_identical = map_lgl(common_names, function(name) {
        !identical(graph$pipeops[[name]]$hash, g2$pipeops[[name]]$hash)
      })
      stopf("Both graphs have PipeOps with ID(s) %s but they don't have the same hashes.",
        paste0("'", common_names[not_identical], "'", collapse = ", ")
      )
    }

    # copy all PipeOps that are in g2 but not in g1
    graph$pipeops = c(graph$pipeops, g2$pipeops[setdiff(names(g2$pipeops), common_names)])

    # clear param_set cache
    graph$.__enclos_env__$private$.param_set = NULL

    # edges that are in md2's graph that were not in md1's graph
    new_edges = g2$edges[!graph$edges, on = c("src_id", "src_channel", "dst_id", "dst_channel")]

    # IDs and channel names that get new input edges. These channels must not already have incoming edges in md1.
    new_input_edges = unique(new_edges[, c("dst_id", "dst_channel"), with = FALSE])

    forbidden_edges = graph$edges[new_input_edges, on = c("dst_id", "dst_channel"), nomatch = NULL]

    if (nrow(forbidden_edges)) {
      stopf("PipeOp(s) %s have differing incoming edges in g1 and g2",
        paste(forbidden_edges$dst_id, collapse = ", "))

    }
    graph$edges = rbind(graph$edges, new_edges)
  }

  return(graph)
}
