test_that("PipeOpModule: basic checks", {
  # nn_Module cannot be used (otherwise the phash would be wrong)
  expect_error(po("module", module = attr(nn_relu(), "module")))

  po_fn = po("module", id = "identity", module = identity, packages = "R6", inname = "a", outname = "b")
  expect_pipeop(po_fn)
  expect_equal(po_fn$id, "identity")
  expect_equal(po_fn$input$name, "a")
  expect_equal(po_fn$output$name, "b")
  po_nn = po("module", id = "relu", module = nn_relu())
  expect_pipeop(po_nn)
  expect_equal(po_nn$id, "relu")
  expect_true("torch" %in% po_nn$packages)
  expect_true("mlr3torch" %in% po_nn$packages)
})

test_that("PipeOpModule works", {
  po_module = PipeOpModule$new("linear", torch::nn_linear(10, 20), inname = "abc", outname = "xyz")
  x = torch::torch_randn(16, 10)
  # This calls the forward function of the wrapped module.
  y_po = with_no_grad(po_module$train(list(input = x)))
  y = with_no_grad(po_module$module(x))
  expect_list(y_po, types = "torch_tensor")
  expect_true(names(y_po) == "xyz")
  expect_true(torch_equal(y, y_po[[1L]]))

  # multiple input and output channels
  nn_custom = torch::nn_module("nn_custom",
    initialize = function(in_features, out_features) {
      self$lin1 = torch::nn_linear(in_features, out_features)
      self$lin2 = torch::nn_linear(in_features, out_features)
    },
    forward = function(x, z) {
      list(out1 = self$lin1(x), out2 = torch::nnf_relu(self$lin2(z)))
    }
  )
  d = 3

  module = nn_custom(d, 2)
  po_module = PipeOpModule$new("custom", module, inname = c("x", "z"), outname = c("out1", "out2"))
  input = list(x = torch_randn(2, d), z = torch_randn(2, d))
  y = po_module$train(input)
  expect_true(all(sort(names(y)) == c("out1", "out2")))
  expect_list(y, types = "torch_tensor")
})

test_that("Cloning works", {
  po_nn = po("module", module = nn_linear(1, 1))$clone(deep = TRUE)
  po_nn1 = po_nn$clone(deep = TRUE)

  # phash should and is different between hashes
  unlockBinding(".additional_phash_input", get_private(po_nn))
  unlockBinding(".additional_phash_input", get_private(po_nn1))
  get_private(po_nn, ".additional_phash_input") = function(...) NULL
  get_private(po_nn1, ".additional_phash_input") = function(...) NULL

  expect_deep_clone_mlr3torch(po_nn, po_nn1)

  po_fn = po("module", module = function(x) x + 1)
  po_fn1 = po_fn$clone(deep = TRUE)
  expect_deep_clone_mlr3torch(po_fn, po_fn1)
})

test_that("phash for PipeOpModule works", {
  f = function(x) x

  fc = compiler::cmpfun(f)

  fe = f
  environment(fe) = new.env()

  p1 = po("module", module = f)
  p2 = po("module", module = fc)
  p3 = po("module", module = fe)

  expect_equal(p1$phash, p2$phash)
  # `f` reads nothing from its environment, so re-homing it does not change what it is
  expect_equal(p1$phash, p3$phash)

  # two modules with the same class and the same parameter shapes are the same configuration; the
  # parameter *values* are state and stay out of `$phash`, as they do for every other PipeOp
  f1 = po("module", module = nn_linear(1, 1))
  f2 = po("module", module = nn_linear(1, 1))
  expect_equal(f1$phash, f2$phash)
  expect_false(f1$phash == po("module", module = nn_linear(2, 3))$phash)
})

test_that("phash of a PipeOpModule does not depend on the module's identity in memory", {
  # `$phash` used to be `address(self$module)` for an `nn_module` and `address(environment(fn))` for
  # a plain function, so a copy of a PipeOpModule -- a deep clone, or one that travelled through
  # `serialize()` -- did not hash like the original.
  po_module = po("module", module = nn_linear(3, 4))
  expect_equal(po_module$phash, po_module$clone(deep = TRUE)$phash)
  expect_equal(po_module$phash, unserialize(serialize(po_module, NULL))$phash)

  # `nn_graph()`'s deep clone detaches `$module` before re-attaching the copy, so a `$phash` that
  # recomputed from `$module` would depend on when it was asked
  graph_module = po_module$clone(deep = TRUE)
  graph_module$module = NULL
  expect_equal(po_module$phash, graph_module$phash)
})

test_that("phash of a PipeOpModule tells modules apart", {
  expect_false(po("module", module = nn_relu())$phash == po("module", module = nn_sigmoid())$phash)
  # `nn_relu()` and `nn_sigmoid()` have no parameters at all, so the class has to carry them apart;
  # for two modules of one class it is the parameter shapes that differ
  expect_false(po("module", module = nn_linear(3, 4))$phash == po("module", module = nn_linear(3, 5))$phash)
  expect_false(po("module", module = function(x) x)$phash == po("module", module = function(x) x + 1)$phash)

  # a closure is told apart by what it reads from its environment, not by which environment that is
  mk = function(a) po("module", module = mlr3misc::crate(function(x) x + a, a))
  expect_equal(mk(1)$phash, mk(1)$phash)
  expect_false(mk(1)$phash == mk(2)$phash)
})

test_that("phash of a PipeOpModule is reproducible across R sessions", {
  skip_on_cran()
  skip_if_not_installed("callr")
  # `address()` differs between processes, so a hash built from it never was reproducible
  pkg = if (requireNamespace("pkgload", quietly = TRUE) &&
      isTRUE(pkgload::is_dev_package("mlr3torch"))) pkgload::pkg_path() else NULL
  in_session = function() {
    callr::r(function(pkg) {
      if (is.null(pkg)) suppressMessages(library(mlr3torch)) else pkgload::load_all(pkg, quiet = TRUE)
      c(
        module = po("module", module = torch::nn_linear(3, 4))$phash,
        fn = po("module", module = mlr3misc::crate(function(x) x + a, a = 1))$phash
      )
    }, args = list(pkg = pkg), libpath = .libPaths())
  }
  expect_equal(in_session(), in_session())
})
