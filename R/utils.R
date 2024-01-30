#' Auto Device
#'
#' First tries cuda, then mp, then cpu.
#'
#' @param device (`character(1)`)\cr
#'   The device. If not `NULL`, is returned as is.
#' @export
auto_device = function(device = NULL) {
  if (device == "auto") {
    device = if (cuda_is_available()) {
      "cuda"
    } else if (backends_mps_is_available()) {
      "mps"
    } else {
      "cpu"
    }
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

uniqueify = function(new, existing) {
  make.unique(c(existing, new), sep = "_")[length(existing) + seq_along(new)]
}

shape_to_str = function(x) {
  shapedescs = map_chr(x, function(y) paste0("(", paste(y, collapse = ",", recycle0 = TRUE), ")"))
  if (test_named(x)) {
    repr = paste0("[", names(x), ": ",  paste(shapedescs, collapse = ";", recycle0 = TRUE), "]")
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
    list(x = torchvision::transform_to_tensor(magick::image_read(self$uris[x])))
  },
  .length = function() {
    length(self$uris)
  }
)

list_to_batch = function(tensors) {
  torch_cat(map(tensors, function(x) x$unsqueeze(1)), dim = 1L)
}

auto_cache_lazy_tensors = function(lts) {
  any(duplicated(map_chr(lts, function(x) dd(x)$dataset_hash)))
}
