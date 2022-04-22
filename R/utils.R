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
  dl = as_dataloader(task, batch_size = batch_size, device = "cpu")
  batch = dl$.iter()$.next()
  y = batch$y
  return(batch)
}

get_instance = function(task) {
  # TODO: Change this to "meta"
  data_loader = as_dataloader(task, batch_size = 1, device = "cpu")
  instance = data_loader$.iter()$.next()
  return(instance)
}

is_tabular = function(task) {
  test("imageuri" %nin% task$features)
}

get_optimizer = function(name) {
  getFromNamespace(sprintf("optim_%s", name), ns = "torch")
}

get_loss = function(name) {
  getFromNamespace(sprintf("nn_%s_loss", name), ns = "torch")
}

get_activation = function(name) {
  assert_choice(name, torch_reflections$activation)
  getFromNamespace(sprintf("nn_%s", name), ns = "torch")
}

assert_optimizer = function(x) {
  assert_true(class(attr(x, "Optimizer")) == "R6ClassGenerator")
}

assert_loss = function(x) {
  assert_true(inherits(x, "nn_loss"))
}

get_cache_dir = function(cache) {
  if (isFALSE(cache)) {
    return(FALSE)
  }
  if (isTRUE(cache)) {
    cache = R_user_dir("mlr3torch", "cache")
  }
  assert(check_directory_exists(cache), check_path_for_output(cache))
  normalizePath(cache, mustWork = FALSE)
}
