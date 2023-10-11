auto_device = function(device = NULL) {
  if (device == "auto") {
    device = if (cuda_is_available()) "cuda" else "cpu"
    lg$debug("Auto-detected device '%s'.", device)
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


make_check_vector = function(d) {
  crate(function(x) {
    if (is.null(x) || test_integerish(x, any.missing = FALSE) && (length(x) %in% c(1, d))) {
      return(TRUE)
    }
    tmp = if (d == 1) "." else sprintf(" or %s.", d)
    sprintf("Must be an integerish vector of length 1%s", tmp)
    }, d, .parent = topenv())
}

check_function_or_null = function(x) check_function(x, null.ok = TRUE)

check_integerish_or_null = function(x) check_integerish(x, null.ok = TRUE)

broadcast = function(shape1, shape2) {
  assert_true(!anyNA(shape1) && !anyNA(shape2))
  d = abs(length(shape1) - length(shape2))
  if (d != 0) {
    if (length(shape1) > length(shape2)) {
      x = c(rep(1L, d), shape1)
    } else {
      y = c(rep(1L, d), shape2)
    }
  }
  pmax(shape1, shape2)
}

broadcast_list = function(...) {
  Reduce(broadcast, list(...))
}

check_nn_module_generator = function(x) {
  if (inherits(x, "nn_module_generator")) {
    return(TRUE)
  }

  "Most be module generator."
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

get_nout = function(task) {
  switch(task$task_type,
    regr = 1,
    classif = length(task$class_names),
    stopf("Unknown task type '%s'.", task$task_type)
  )
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


# a function that has argument names 'names' and returns its arguments as a named list.
# used to simulate argument matching for `...`-functions.
# example:
# f = argument_matcher(c("a", "b", "c"))
# f(1, 2, 3) --> list(a = 1, b = 2, c = 3)
# f(1, 2, a = 3) --> list(a = 3, b = 1, c = 2)
# usecase:
# ff = function(...) {
#   l = argument_matcher(c("a", "b", "c"))(...)
#   l$a + l$b
# }
# # behaves like
# ff(a, b, c) a + b
# (Except in the aqward case of missing args)
argument_matcher = function(args) {
  fn = as.function(c(named_list(args, substitute()), quote(as.list(environment()))))
  environment(fn) = topenv()
  fn
}

as_shape = function(shape) {
  assert_integerish(shape)
  if (!is.na(shape[1L])) {
    shape = c(NA_integer_, shape)
  }
  assert_integerish(shape[-1], any.missing = FALSE)
  shape
}

assert_shape = function(shape, null.ok = FALSE) { # nolint
  if (is.null(shape) && null.ok) return(TRUE)

  assert_integerish(shape, min.len = 2L)
  if (!(is.na(shape[[1L]]) || anyNA(shape[-1L]))) {
    stopf("Shape must have exactly one NA in the batch dimension.")
  }
  TRUE
}

assert_shapes = function(shapes) {
  assert_list(shapes, names = "unique", min.len = 1L)
  walk(shapes, assert_shape)
}
