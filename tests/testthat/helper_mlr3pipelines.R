library("mlr3pipelines")
library("checkmate")
library("testthat")
library("R6")
library("mlr3misc")
library("paradox")

lapply(list.files(system.file("testthat", package = "mlr3pipelines"), pattern = "^helper.*\\.[rR]", full.names = TRUE), source)

mlr_helpers = list.files(system.file("testthat", package = "mlr3pipelines"), pattern = "^helper.*\\.[rR]", full.names = TRUE)
lapply(mlr_helpers, FUN = source)

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

    if (inherits(a, "nn_module_generator") || inherits(a, "torch_optimizer_generator")) {
      return(NULL)
    }
    if (inherits(a, "R6ClassGenerator")) {
      return(NULL)
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


