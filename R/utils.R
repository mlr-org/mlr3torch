#' Auto Device
#'
#' First tries cuda, then cpu.
#'
#' @param device (`character(1)`)\cr
#'   The device. If not `NULL`, is returned as is.
#' @export
auto_device = function(device = NULL) {
  if (!is.null(device) && device == "auto") {
    device = if (cuda_is_available()) "cuda" else "cpu"
    lg$debug("Auto-detected device '%s'.", device)
  }
  if (!is.null(device) && device == "cuda" && !cuda_is_available()) {
    stopf("Device is set to 'cuda', but no CUDA device is available. Set `device` to 'cpu', or to 'auto' to select automatically.") # nolint
  }
  return(device)
}

running_on_mac = function() {
  Sys.info()["sysname"] == "Darwin"
}

inferps = function(fn, ignore = character(0), tags = "train") {
  if (inherits(fn, "R6ClassGenerator")) {
    fn = get_init(fn)
    if (is.null(fn)) {
      return(ps())
    }
  }
  assert_function(fn)
  assert_character(ignore, any.missing = FALSE)
  ignore = union(ignore, "...")
  frm = formals(fn)
  frm = frm[names(frm) %nin% ignore]

  frm_domains = lapply(frm, function(formal) p_uty(tags = tags))

  do.call(paradox::ps, frm_domains)
}


# `null_ok` must be `FALSE` for parameters the operator cannot do without
make_check_vector = function(d, null_ok = TRUE) {
  crate(function(x) {
    if ((null_ok && is.null(x)) || test_integerish(x, any.missing = FALSE) && (length(x) %in% c(1, d))) { # nolint
      return(TRUE)
    }
    tmp = if (d == 1) "." else sprintf(" or %s.", d)
    sprintf("Must be an integerish vector of length 1%s", tmp)
    }, d, null_ok, .parent = topenv())
}

assert_inherits_classname = function(class_generator, classname) {
  assert_class(class_generator, "R6ClassGenerator")
  while (!is.null(class_generator)) {
    if (class_generator$classname == classname) {
      return(TRUE)
    }
    class_generator = class_generator$get_inherit()
  }
  stopf("R6ClassGenerator does not generate object that inherits from %s.", classname)
}

get_init = function(x) {
  cls = class_with_init(x)
  if (is.null(cls)) return(NULL)
  cls$public_methods$initialize
}

# jarl-ignore unused_function: called from man-roxygen/learner_example.R, which jarl does not scan
default_task_id = function(learner) {
  task_id = get0("task_id", envir = parent.frame(), inherits = FALSE)
  if (!is.null(task_id)) {
    return(task_id)
  }
  if (inherits(learner, "LearnerTorchImage")) {
    stopf("Currently not available!")
  }
  switch(learner$task_type,
    classif = "iris",
    regr = "mtcars",
    stopf("No default task type.")
  )

}

class_with_init = function(x) {
  if (is.null(x)) {
    # This is the case where no initialize method is found
    return(NULL)
  } else if (is.null(x$public_methods) || exists("initialize", x$public_methods, inherits = FALSE)) {
    return(x)
  } else {
    Recall(x$get_inherit())
  }
}

sample_input_from_shapes = function(shapes, n = 1L) {
  assert_list(shapes, types = "numeric", min.len = 1)
  assert_int(n)
  imap(shapes, function(shape, nm) {
    shape[1] = n
    invoke(torch_randn, .args = as.list(shape))
  })
}

load_col_info = function(name) {
  readRDS(system.file("col_info", paste0(name, ".rds"), package = "mlr3torch"))
}


test_equal_col_info = function(x, y) {
  nms = c("id", "type", "levels")
  if (!(test_permutation(colnames(x), nms) && test_permutation(colnames(y), nms))) {
    return(FALSE)
  }

  x = x[order(get("id"))]
  y = y[order(get("id"))]

  isTRUE(all.equal(x$id, y$id)) && isTRUE(all.equal(x$type, y$type)) &&
    all(pmap_lgl(list(x = x$levels, y = y$levels), function(x, y) isTRUE(all.equal(x, y))))
}


uniqueify = function(new, existing) {
  make.unique(c(existing, new), sep = "_")[length(existing) + seq_along(new)]
}

#' @rdname shape_helpers
#' @export
shape_to_str = function(x) {
  if (is.numeric(x) || is.logical(x)) { # single shape
    return(sprintf("(%s)", paste0(x, collapse = ",")))
  }
  assert(test_list(x) || is.null(x))
  if (is.null(x)) {
    return("(<unknown>)")
  }

  shapedescs = map_chr(x, function(y) {
    if (is.null(y)) {
      return("<unknown>")
    }
    paste0("(", paste(y, collapse = ",", recycle0 = TRUE), ")")
  })
  if (test_named(x)) {
    repr = paste0("[", paste(paste0(names(x), ": ", shapedescs), collapse = "; ", recycle0 = TRUE), "]")
    return(repr)
  }
  paste0("[",  paste(shapedescs, collapse = ";", recycle0 = TRUE), "]")
}

test_equal = function(x, y) {
  isTRUE(all.equal(x, y))
}


dataset_image = dataset("image_dataset",
  initialize = function(uris) {
    self$uris = uris
  },
  .getitem = function(x) {
    list(x = torchvision::transform_to_tensor(torchvision::base_loader(self$uris[x])))
  },
  .length = function() {
    length(self$uris)
  }
)

list_to_batch = function(tensors) {
  torch_cat(map(tensors, function(x) x$unsqueeze(1)), dim = 1L)
}

auto_cache_lazy_tensors = function(lts) {
  if (length(lts) <= 1L) {
    return(FALSE)
  }
  anyDuplicated(unlist(map_if(lts, function(x) length(x) > 0, function(x) dd(x)$dataset_hash))) > 0L
}

#' Replace the head of a network
#' Replaces the head of the network with a linear layer with d_out classes.
#' @param network ([`torch::nn_module`])\cr
#'   The network
#' @param d_out (`integer(1)`)\cr
#'   The number of output classes.
#' @export
#' @keywords internal
replace_head = function(network, d_out) {
  UseMethod("replace_head")
}

# Creates a check function (as expected by `custom_check`) that asserts that `x` inherits from
# `classes`. Named `make_check_class()` and not `check_class()` to avoid masking
# `checkmate::check_class()`.
make_check_class = function(classes) {
  assert_character(classes, any.missing = FALSE, min.len = 1L)
  crate(function(x) {
    check_class(x, classes)
  }, classes, .parent = topenv())
}

check_nn_module = function(x) {
  check_class(x, "nn_module")
}

check_nn_module_generator = function(x) {
  check_class(x, "nn_module_generator")
}

LossNone = function() {
  structure(list(), class = "LossNone")
}

OptimizerNone = function() {
  structure(list(), class = "OptimizerNone")
}

CallbacksNone = function() {
  structure(list(), class = "CallbacksNone")
}

get_example_batch = function(dl) {
  ds = dl$dataset
  if (length(ds) < 2) {
    stopf("Dataset needs to contain at least 2 observations")
  }
  if (!is.null(ds$.getbatch)) {
    ds$.getbatch(1:2)
  } else {
    torch_stack(list(
      ds$.getitem(1),
      ds$.getitem(2)
    ))
  }
}

# l: list containing args
# f: function to be called
# returns: l reordered so it can be passed by position
order_named_args = function(f, l) {
  args = formalArgs(f)
  x = match("...", args)
  f2 = f
  body(f2) = quote(as.list(match.call())[-1L])
  l2 = do.call(f2, l)
  # (function(..., x = 1) {})(1, 2, 3) works
  # (function(..., x = 1) {})(1, 2, x = 3) DOES NOT WORK
  if (!is.null(names(l2)) && !is.na(x) && x != length(args)) {
    stopf("Because arguments are passed to tracer by position, `...` must either be the only or the last argument when named arguments are also passed.")
  }
  l2
}



#' @title Network Output Dimension
#' @description
#' Calculates the output dimension of a neural network for a given task that is expected by
#' \pkg{mlr3torch}.
#' For classification, this is the number of classes (unless it is a binary classification task,
#' where it is 1). For regression, it is 1.
#'
#' This is an S3 generic and the single place where \pkg{mlr3torch} decides how many output neurons
#' a task needs: it is what [`PipeOpTorchHead`] and the [`LearnerTorch`]s that build their own head
#' ask. Adding a method for a new task type is therefore the way to support it, see the
#' "Supporting Other Task Types" section of [`PipeOpTorchHead`].
#'
#' @param x (any)\cr
#'   The task.
#' @param ... (any)\cr
#'   Additional arguments. Not used yet.
#' @return (`integer(1)`) The number of output neurons.
#' @seealso [`PipeOpTorchHead`]
#' @export
output_dim_for = function(x, ...) {
  UseMethod("output_dim_for")
}

#' @export
output_dim_for.TaskClassif = function(x, ...) {
  if ("twoclass" %in% x$properties) {
    return(1L)
  }
  length(x$class_names)
}

#' @export
output_dim_for.TaskRegr = function(x, ...) {
  1L
}

single_lazy_tensor = function(task) {
  identical(task$feature_types[, "type"][[1L]], "lazy_tensor")
}

n_num_features = function(task) {
  sum(task$feature_types$type %in% c("numeric", "integer"))
}

n_categ_features = function(task) {
  sum(task$feature_types$type %in% c("factor", "ordered", "logical"))
}

# Cardinalities of the categorical features of a task, in the column order that
# `ingress_categ()` produces.
# Two things this must get right and that are easy to get wrong:
#  * `Task$levels()` returns `NULL` for `logical()` features, so their cardinality has to be
#    supplied explicitly (it is always 2). Taking `lengths(task$levels(...))` alone yields 0.
#  * `task$feature_names` and `task$feature_types` are not always in the same order (e.g. after
#    `po("scale")`), so the feature order must come from the ingress token, not from
#    `task$feature_names`. Otherwise the cardinalities silently desync from the columns.
categ_cardinalities = function(task) {
  features = ingress_categ()$features(task)
  if (!length(features)) {
    return(integer(0))
  }
  types = task$feature_types[list(features), "type", on = "id"][[1L]]
  cardinalities = lengths(task$levels(features))[features]
  cardinalities[types == "logical"] = 2L
  set_names(as.integer(cardinalities), features)
}

# Identity of a `function()`, including the values that it captures.
#
# `mlr3misc::hash_input.function()` reads `formals()` and `as.character(body())` -- the AST, never
# the byte code that R's JIT compiler installs on first call -- and is therefore stable across
# copies and sessions. It ignores the closure's environment, which is not enough here:
# `PipeOpTaskPreprocTorch` wraps every preprocessing function into a `crate()`d closure whose body
# is the same for all of them, and which is told apart only by the `trafo` and `param_vals` that it
# captures. Dropping the environment would make all of them collide, and their identity ends up in
# `DataDescriptor$hash`, i.e. in the key of the materialization cache.
#
# Environments that have a name -- namespaces, the global environment, package environments --
# stand for themselves: walking their contents would be expensive, and they are not part of what a
# closure "is". Anonymous environments, which is what `crate()` creates, are hashed by content.
hash_input_closure = function(x) {
  list(hash_input(x), hash_input_environment(environment(x)))
}

# Content of an environment, with `seen` guarding against the reference cycles that R6 objects
# (`self`, `private`, `super`) and recursive closures introduce.
hash_input_environment = function(env, seen = list(), depth = 0L) {
  if (!is.environment(env)) {
    return(NULL)
  }
  name = environmentName(env)
  if (nzchar(name)) {
    return(name)
  }
  if (depth >= 20L || some(seen, function(e) identical(e, env))) {
    return("<not descended into>")
  }
  seen = c(seen, env)
  values = as.list(env, all.names = TRUE)
  values = values[order(names(values))]
  list(names(values), map(values, hash_input_value, seen = seen, depth = depth + 1L),
    hash_input_environment(parent.env(env), seen, depth + 1L))
}

hash_input_value = function(x, seen = list(), depth = 0L) {
  if (depth >= 20L) {
    return("<not descended into>")
  }
  if (inherits(x, "nn_module")) {
    return(nn_module_identity(x, seen, depth))
  }
  if (is.function(x)) {
    return(list(hash_input(x), hash_input_environment(environment(x), seen, depth + 1L)))
  }
  if (is.environment(x)) {
    return(hash_input_environment(x, seen, depth + 1L))
  }
  if (is.list(x)) {
    return(map(x, hash_input_value, seen = seen, depth = depth + 1L))
  }
  hash_input(x)
}

# Identity of an `nn_module` *instance*: its class, the values it was configured with, the shapes
# and dtypes of its parameters and buffers, and the same for its sub-modules.
#
# This used to be `data.table::address()`, which changes whenever the module is copied. Since
# `PipeOpModule$deep_clone()` copies it, a deep clone hashed differently from its original.
#
# The parameter *values* are deliberately left out.
#  * They are not free: reading them means pulling every weight out of torch and into R, whereas
#    everything hashed here is metadata.
#  * `PipeOpTorchModel` re-initializes the whole network from the learner's seed before training
#    (`.reset_parameters_`), so the weights that a module happens to carry while the graph is
#    assembled do not influence the fitted model.
#  * Leaving them out is what makes two identically specified networks hash identically, which is
#    the point of hashing them at all.
# The price is that a module whose configuration shows up *only* in the values of its parameters is
# not told apart from another one -- `torch::nn_prelu(init = )` is the single such case among the
# modules that this package builds.
#
# Methods are hashed via `hash_input()` (formals and deparsed body) rather than by `digest()`ing
# them, again to stay clear of the byte code. They are not redundant with the class: a module
# created by `nn_module()` without a `classname` has class `"nn_module"` and nothing else.
nn_module_identity = function(x, seen = list(), depth = 0L) {
  if (inherits(x, "nn_module_generator")) {
    # `attr(x, "module")` is an `R6ClassGenerator` then, not an instance
    return(hash_input(x))
  }
  instance = attr(x, "module")
  if (!R6::is.R6(instance)) {
    return(class(x))
  }
  # `get()` rather than `instance$.__enclos_env__`: torch defines a `$.nn_Module` that does not
  # hand out the enclosing environment itself, so the cycle check below would never fire
  enclos = get(".__enclos_env__", envir = instance)
  if (depth >= 20L || some(seen, function(e) identical(e, enclos))) {
    return("<not descended into>")
  }
  seen = c(seen, enclos, instance)
  fields = list()
  for (name in ls(instance)) {
    # active bindings are torch's views on the parameters, buffers and sub-modules, which are
    # covered by `state_dict()` and `children` below
    if (bindingIsActive(name, instance)) next
    value = get(name, envir = instance)
    fields[[name]] = if (is.function(value) && !inherits(value, "nn_module") &&
      identical(environment(value), enclos)) {
      # a method of this class: it is enclosed by the very object being described here, so only its
      # code carries information. Functions that a module *stores* (e.g. what `nn_fn` wraps) are not
      # enclosed by it and go through `hash_input_value()`, which keeps what they capture.
      hash_input(value)
    } else {
      hash_input_value(value, seen, depth + 1L)
    }
  }
  state = map(instance$state_dict(), function(tensor) list(dim(tensor), as.character(tensor$dtype)))
  children = map(instance$children, nn_module_identity, seen = seen, depth = depth + 1L)
  list(class(x), names(fields), fields, names(state), state, names(children), children)
}

# Identity of what a `PipeOpModule` wraps, which is either an `nn_module` or a plain function.
module_identity = function(x) {
  hash_input_value(x)
}
