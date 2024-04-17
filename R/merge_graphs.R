merge_graphs = function(g1, g2) {
  graph = g1$clone(deep = FALSE)

  # if graphs are identical, we don't need to worry about copying stuff
  if (!identical(g1, g2)) {
    # PipeOps that have the same ID that occur in both graphs must be identical.
    common_names = intersect(names(graph$pipeops), names(g2$pipeops))
    if (!identical(graph$pipeops[common_names], g2$pipeops[common_names])) {
      not_identical = map_lgl(common_names, function(name) {
        !identical(graph$pipeops[[name]], g2$pipeops[[name]])
      })
      stopf("Both graphs have PipeOps with ID(s) %s but they are not identical.",
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
