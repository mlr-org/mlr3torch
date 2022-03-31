#' Extracts the parametersts from various grpahs
extract_paramset = function(graphs) {
  psn = ps() # param set new
  imap(graphs,
    function(graph, name) {
      pvals = graph$param_set$values
      map(graph$param_set$params,
        function(param) {
          pido = param$id # param id old
          pidn = sprintf("%s.%s", name, param$id)
          param$id = pidn
          psn$add(param$clone())
          if (pido %in% names(pvals)) {
            psn$values = c(psn$values, set_names(pvals[[pido]], pidn))
          }
        }
      )
    }
  )
  return(psn)
}

get_orphan = function(nodes) {
  orphans = nodes[["id"]][map_lgl(nodes[["parents"]], function(x) !length(x))]
  assert_true(length(orphans) == 1L)
  return(orphans)
}

is_tokenizer = function(x) {
  assert_true("TorchOpTokenizer" %in% attr(orphan, "TorchOp"))
}

# Mostly used in the tests to get a single batch from a task that can be fed into a network
make_batch = function(task, batch_size) {
  dl = make_dataloader(task, batch_size = batch_size, device = "cpu")
  batch = dl$.iter()$.next()
  y = batch$y
  batch$y = NULL
  return(batch)
}

get_instance = function(task) {
  #' TODO: Change this to "meta"
  data_loader = make_dataloader(task, 1, "cpu")
  instance = data_loader$.iter()$.next()
  return(instance)
}
