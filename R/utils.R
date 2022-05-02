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

# Mostly used in the tests to get a single batch from a task that can be fed into a network
get_batch = function(x, batch_size, device) {
  UseMethod("get_batch")
}

get_batch.dataset = function(x, batch_size, device) {
  get_batch(as_dataloader(x, device = device, batch_size = batch_size))
}

get_batch.dataloader = function(x, ...) {
  # TODO: Change this to "meta"
  instance = x$.iter()$.next()
}

get_batch.Task = function(x, batch_size, device) {
  get_batch(as_dataloader(x, batch_size = batch_size, device = device))
}

.S3method("get_batch", "dataloader", get_batch.dataloader)
.S3method("get_batch", "dataset", get_batch.dataset)
.S3method("get_batch", "Task", get_batch.Task)

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

get_image_trafo = function(trafo) {
  torch_trafo = getFromNamespace(x = sprintf("transform_%s", trafo), ns = "torchvision")
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
