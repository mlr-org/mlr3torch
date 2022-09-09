# Mostly used in the tests to get a single batch from a task that can be fed into a network
get_batch = function(x, batch_size, device) {
  UseMethod("get_batch")
}

get_batch.dataset = function(x, batch_size, device) { # nolint
  get_batch(as_dataloader(x, device = device, batch_size = batch_size))
}

get_batch.dataloader = function(x, ...) { # nolint
  # TODO: Change this to "meta"
  instance = x$.iter()$.next()
}

get_batch.Task = function(x, batch_size, device) { # nolint
  get_batch(as_dataloader(x, batch_size = batch_size, device = device))
}

.S3method("get_batch", "dataloader", get_batch.dataloader)
.S3method("get_batch", "dataset", get_batch.dataset)
.S3method("get_batch", "Task", get_batch.Task)

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

get_normalization = function(name) {
  assert_choice(name, torch_reflections$normalization)
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

# We need this to avoid name clashes in PipeOpFFE
uniqueify = function(name, existing, count = 0L) {
  # Special case that covers 99.99 % of the cases in the PipeOp above
  if ((count == 0L) && name %nin% existing) {
    return(name)
  }

  if (count == 100L) {
    stopf("Choose a better name.")
  }

  count_inc = count + 1L
  alternative = sprintf("%s_%s", name, count_inc)
  if (alternative %nin% existing) {
    return(alternative)
  } else {
    uniqueify(name, existing, count_inc)
  }
}

is_torchop = function(x) {
  inherits(x, "TorchOp")
}

is_graph = function(x) {
  inherits(x, "Graph")
}

get_measure = function(x) {
  getFromNamespace(x, ns = "mlr3measures")
}

get_measures = function(xs) {
  map(xs, get_measure)
}

#' @title Splits a list based on names
#'
#' @description
#' Named lists can be split into multiple lists by applying regexes to its names.
#' If the list is unnamed, all names are treated to be `""` for consistency.
#'
#' @param x (named `list()`)\cr
#'   The list that will be split.
#' @param patterns (`character()`)\cr
#'   A list containing various regex patterns. If it is named, the output inherits those names.
#' @param ... (any)\cr
#'   Additional arguments to `grepl()`.
#'
#' @return
#' A (possibly named) `list()` of subsets of x.
#'
#' @export
split_list = function(x, patterns, ...) {
  assert_list(x)
  assert_character(patterns, any.missing = FALSE)
  if (length(x)) {
    nms = names(x) %??% ""
    out = map(
      patterns,
      function(pattern) {
        x[grepl(pattern, nms, ...)]
      }
    )
  } else {
    out = replicate(length(patterns), list())
  }

  set_names(out, names(patterns))
}

freeze_params = function(model) {
  for (par in model$parameters) {
    par$requires_grad_(FALSE)
  }
}

dbuild = function(op) {
  debugonce(op$.__enclos_env__$.build)
}

make_named_returns = function(f) {
  assert_class(f, "nn_module")
  g = function() {

  }
}

include_template = function(x) {
  paste(readLines(system.file("templates", paste0(x, ".md"), package = "mlr3torch")), collapse = "\n")
}

