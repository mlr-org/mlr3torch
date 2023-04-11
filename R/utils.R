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


check_measures = function(x) {
  if (!is.list(x)) {
    x = list(x)
  }
  if (test_list(x, types = "Measure")) {
    ids = map_chr(x, "id")
    if (test_names(ids, type = "unique")) {
      return(TRUE)
    }
  }

  "Parameter must be a Measure or list of Measures with valid ids."
}

check_network = function(x) {
  if (inherits(x, "nn_Module")) {
    "The network must be initialized by calling the function (and not with '$new()')."
  } else if (!test_class(x, "nn_module")) {
    "Must be a 'nn_module()'."
  } else {
    TRUE
  }
}

check_vector = function(d) {
  function(x) {
    if (is.null(x) || test_integerish(x, any.missing = FALSE) && (length(x) %in% c(1, d))) {
      return(TRUE)
    }
    sprintf("Must be an integerish vector of length 1 or %s", d)
  }
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


get_init = function(x) {
  cls = class_with_init(x)
  if (is.null(cls)) return(NULL)
  cls$public_methods$initialize
}

class_with_init = function(x) {
  if (is.null(x)) {
    # This is the case where no initialize method is found
    return(NULL)
  } else if (!is.null(x$public_methods$initialize)) {
    return(x)
  } else {
    Recall(x$get_inherit())
  }
}



with_seed = function(seed, expr) {
  old_seed = get0(".Random.seed", globalenv(), mode = "integer", inherits = FALSE)
  if (is.null(old_seed)) {
    runif(1L)
    old_seed = get0(".Random.seed", globalenv(), mode = "integer", inherits = FALSE)
  }

  on.exit(assign(".Random.seed", old_seed, globalenv()), add = TRUE)
  set.seed(seed)
  force(expr)
}

batch_from_task = function(task, batch_size, device, learner, param_vals = NULL) {
  get_private(learner)$.dataloader(task, param_vals %??% learner$param_set$values)
}
