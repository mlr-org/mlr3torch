auto_device = function(device = NULL) {
  if (device == "auto") {
    device = if (cuda_is_available()) "cuda" else "cpu"
    lg$debug("Auto-detected device '%s'.", device)
  }
  return(device)
}

test_all = function(x, y) {
  test_true(all(x == y))
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
    }, d = d, .parent = topenv())
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

assert_shape = function(shape, name) {
  if (!(is.na(shape[[1L]]) && sum(is.na(shape)) == 1)) {
    stopf("Input '%s' has shape (%s) but must have exactly one NA in the first dimension (batch size).",
      name, paste0(shape, collapse = ", "))
  }
}

assert_shapes = function(shapes) {
  iwalk(shapes, function(shape, name) assert_shape(shape, name))
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
