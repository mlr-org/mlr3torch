build_tabular_resnet = function(self, block, task) {
  d_in = nrow(task$feature_types)
  n_classes = length(task$class_names)
  p = self$param_set$get_values(tag = "train")

  block_params = p[names(p) %in% block$param_set$ids()]
  block$param_set$values = insert_named(
    block$param_set$values, block_params
  )

  normalization = switch(p$normalization,
    batch_norm = top("batch_norm"),
    stopf("Not implemented yet.")
  )

  block$id = "block"

  graph = top("input") %>>%
    top
    block %>>%
    normalization %>>%
    top("output")

  browser()
  graphitecture = graph$train(task)[[1L]]$architecture
  network = graphitecture$build(task)
  return(network)
}

