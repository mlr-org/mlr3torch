expect_po_ingress = function(po_ingress, task) {
  testthat::expect_error(po_ingress$train(list(task))[[1]], regexp = "Task contains features of type")

  task = po("select", selector = selector_type(c(po_ingress$feature_types)))$train(list(task))[[1L]]
  token = po_ingress$train(list(task))[[1L]]

  testthat::expect_true(token$graph$ids() == po_ingress$id)
  testthat::expect_true(all(token$task$feature_types$type %in% po_ingress$feature_types))
  testthat::expect_equal(token$callbacks, named_list())
  testthat::expect_equal(token$.pointer, c(po_ingress$id, "output"))
  testthat::expect_equal(token$.pointer_shape, get_private(po_ingress)$.shape(task, po_ingress$param_set$values))

  ingress = token$ingress
  expect_set_equal(
    ingress[[1L]]$features,
    task$feature_types[get("type") %in% po_ingress$feature_types, "id", with = FALSE][[1L]]
  )

  ds = task_dataset(task, ingress, device = "cpu")
  batch = ds$.getbatch(1)
  x = batch$x[[1L]]
  testthat::expect_true(torch_equal(x, token$ingress[[1L]]$batchgetter(task$data(1, token$task$feature_names), "cpu")))
}

expect_man_exists = function(man) {
  checkmate::expect_string(man, na.ok = TRUE, fixed = "::", label = man)
  if (!is.na(man)) {
    parts = strsplit(man, "::", fixed = TRUE)[[1L]]
    matches = help.search(parts[2L], package = parts[1L], ignore.case = FALSE)
    checkmate::expect_data_frame(matches$matches, min.rows = 1L, info = "man page lookup", label = man)
  }
}

# expect that 'one' is a deep clone of 'two'
expect_deep_clone = function(one, two) {
  # is equal
  expect_equal(one, two)
  visited = new.env()
  visited_b = new.env()
  expect_references_differ = function(a, b, path) {

    force(path)
    if (length(path) > 400) {
      stop("Recursion too deep in expect_deep_clone()")
    }

    # don't go in circles
    addr_a = data.table::address(a)
    addr_b = data.table::address(b)
    if (!is.null(visited[[addr_a]])) {
      return(invisible(NULL))
    }
    visited[[addr_a]] = path
    visited_b[[addr_b]] = path

    # TODO: Take care of torch parameters tensors and buffers and torch optimizer and nn module etc.

    if (inherits(a, "nn_module_generator") || inherits(a, "torch_optimizer_generator") ||
      inherits(a, "R6ClassGenerator")) {
      return(invisible(NULL))
    }

    # follow attributes, even for non-recursive objects
    if (utils::tail(path, 1) != "[attributes]" && !is.null(base::attributes(a))) {
      expect_references_differ(base::attributes(a), base::attributes(b), c(path, "[attributes]"))
    }

    # don't recurse if there is nowhere to go
    if (!base::is.recursive(a)) {
      return(invisible(NULL))
    }

    # check that environments differ
    if (base::is.environment(a)) {
      # some special environments
      if (identical(a, baseenv()) || identical(a, globalenv()) || identical(a, emptyenv())) {
        return(invisible(NULL))
      }
      if (length(path) > 1 && R6::is.R6(a) && "clone" %nin% names(a)) {
        return(invisible(NULL))  # don't check if smth is not cloneable
      }
      if (identical(utils::tail(path, 1), c("[element train_task] 'train_task'"))) {
        return(invisible(NULL))  # workaround for https://github.com/mlr-org/mlr3/issues/382
      }
      if (identical(utils::tail(path, 1), c("[element fallback] 'fallback'"))) {
        return(invisible(NULL))  # workaround for https://github.com/mlr-org/mlr3/issues/511
      }
      label = sprintf("Object addresses differ at path %s", paste0(path, collapse = "->"))
      expect_true(addr_a != addr_b, label = label)
      expect_null(visited_b[[addr_a]], label = label)
    } else {
      a = unclass(a)
      b = unclass(b)
    }

    # recurse
    if (base::is.function(a)) {
      return(invisible(NULL))
      ## # maybe this is overdoing it
      ## expect_references_differ(base::formals(a), base::formals(b), c(path, "[function args]"))
      ## expect_references_differ(base::body(a), base::body(b), c(path, "[function body]"))
    }
    objnames = base::names(a)
    if (is.null(objnames) || anyDuplicated(objnames)) {
      index = seq_len(base::length(a))
    } else {
      index = objnames
      if (base::is.environment(a)) {
        index = Filter(function(x) !bindingIsActive(x, a), index)
      }
    }
    for (i in index) {
      if (utils::tail(path, 1) == "[attributes]" && i %in% c("srcref", "srcfile", ".Environment")) next
      expect_references_differ(base::`[[`(a, i), base::`[[`(b, i), c(path, sprintf("[element %s]%s", i,
        if (!is.null(objnames)) sprintf(" '%s'", if (is.character(index)) i else objnames[[i]]) else "")))
    }
  }
  expect_references_differ(one, two, "ROOT")
}

expect_shallow_clone = function(one, two) {
  expect_equal(one, two)
  if (base::is.environment(one)) {
    addr_a = data.table::address(one)
    addr_b = data.table::address(two)
    expect_true(addr_a != addr_b, label = "Objects are shallow clones")
  }
}

# check basic properties of a pipeop object
# - properties / methods as we need them
expect_pipeop = function(po, check_ps_default_values = TRUE) {

  label = sprintf("pipeop '%s'", po$id)
  expect_class(po, "PipeOp", label = label)
  expect_string(po$id, label = label)
  expect_class(po$param_set, "ParamSet", label = label)
  expect_list(po$param_set$values, names = "unique", label = label)
  expect_flag(po$is_trained, label = label)
  expect_output(print(po), "PipeOp:", label = label)
  expect_character(po$packages, any.missing = FALSE, unique = TRUE, label = label)
  expect_function(po$train, nargs = 1)
  expect_function(po$predict, nargs = 1)
  expect_data_table(po$input, any.missing = FALSE)
  expect_names(names(po$input), permutation.of = c("name", "train", "predict"))
  expect_data_table(po$output, any.missing = FALSE)
  expect_names(names(po$output), permutation.of = c("name", "train", "predict"))
  expect_int(po$innum, lower = 1)
  expect_int(po$outnum, lower = 1)
  expect_valid_pipeop_param_set(po, check_ps_default_values = check_ps_default_values)
}

# autotest for the parmset of a pipeop
# - at least one of "train" or "predict" must be in every parameter's tag
# - custom_checks of ParamUty return string on failure
# - either default or values are set; if both, they differ (only if check_ps_default_values = TRUE)
expect_valid_pipeop_param_set = function(po, check_ps_default_values = TRUE) {
  ps = po$param_set
  expect_true(every(ps$tags, function(x) length(intersect(c("train", "predict"), x)) > 0L))

  uties = ps$params[ps$ids("ParamUty")]
  if (length(uties)) {
    test_value = NO_DEF  # custom_checks should fail for NO_DEF
    results = map(uties, function(uty) {
      uty$custom_check(test_value)
    })
    expect_true(all(map_lgl(results, function(result) {
      length(result) == 1L && (is.character(result) || result == TRUE)  # result == TRUE is necessary because default is function(x) TRUE
    })), label = "custom_check returns string on failure")
  }

  if (check_ps_default_values) {
    default = ps$default
    values = ps$values
    default_and_values = intersect(names(default), names(values))
    if (length(default_and_values)) {
      expect_true(all(pmap_lgl(list(default[default_and_values], values[default_and_values]), function(default, value) {
        !identical(default, value)
      })), label = "ParamSet default and values differ")
    }
  }
}

# Thoroughly check that do.call(poclass$new, constargs) creates a pipeop
# - basic properties check (expect_pipeop)
# - deep clone works
# - *_internal checks for classes
# - *_internal handles NO_OP as it should
expect_pipeop_class = function(poclass, constargs = list(), check_ps_default_values = TRUE) {
  skip_on_cran()
  po = do.call(poclass$new, constargs)

  expect_pipeop(po, check_ps_default_values = check_ps_default_values)

  poclone = po$clone(deep = TRUE)
  expect_deep_clone(po, poclone)

  in_nop = rep(list(NO_OP), po$innum)
  in_nonnop = rep(list(NULL), po$innum)
  out_nop = rep(list(NO_OP), po$outnum)
  names(out_nop) = po$output$name

  expect_false(po$is_trained)
  expect_equal(po$train(in_nop), out_nop)
  expect_equal(po$predict(in_nop), out_nop)
  expect_true(is_noop(po$state))
  expect_true(po$is_trained)

  expect_error(po$predict(in_nonnop), "Pipeop .* got NO_OP during train")

  # check again with no_op-trained PO
  expect_pipeop(po, check_ps_default_values = check_ps_default_values)
  poclone = po$clone(deep = TRUE)
  expect_deep_clone(po, poclone)

}

expect_graph = function(g, n_nodes = NULL, n_edges = NULL) {

  expect_class(g, "Graph")
  expect_list(g$pipeops, "PipeOp")
  if (!is.null(n_nodes)) {
    expect_length(g$pipeops, n_nodes)
  }
  expect_character(g$packages, any.missing = FALSE, unique = TRUE)

  expect_data_table(g$edges, any.missing = FALSE)
  if (!is.null(n_edges)) {
    expect_equal(nrow(g$edges), n_edges)
  }

  expect_character(g$hash)

  expect_class(g$param_set, "ParamSet")
  expect_list(g$param_set$values, names = "unique")

  expect_character(g$lhs, any.missing = FALSE)
  expect_character(g$rhs, any.missing = FALSE)

  expect_set_equal(g$ids(), names(g$pipeops))
  expect_set_equal(g$ids(sorted = TRUE), names(g$pipeops))

  expect_flag(g$is_trained)
}
