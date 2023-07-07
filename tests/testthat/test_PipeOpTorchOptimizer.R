test_that("Basic checks", {
  topt = t_opt("sgd")
  topt$packages = union(topt$packages, "utils")
  obj = po("torch_optimizer", optimizer = topt, lr = 0.123)
  task = tsk("iris")
  md = po("torch_ingress_num")$train(list(task))
  mdout = obj$train(md)[[1L]]

  expect_equal(address(obj$param_set), address(get_private(obj)$.optimizer$param_set))

  expect_set_equal(topt$param_set$ids(), obj$param_set$ids())
  expect_subset("utils", obj$packages)
  expect_true(obj$is_trained)
  expect_identical(obj$state, list())
  expect_pipeop(obj)
  expect_class(mdout$optimizer, "TorchOptimizer")
  expect_class(mdout$optimizer$generator, "optim_sgd")
  expect_true(mdout$optimizer$param_set$values$lr == 0.123)

  expect_error(po("torch_optimizer", list()))
})

test_that("Correct error message if optimizer is already configured", {
  task = tsk("iris")
  md = po("torch_ingress_num")$train(list(task))

  obj = po("torch_optimizer", "adam")
  mdout = obj$train(md)
  expect_error(obj$train(mdout),
    regexp = "The optimizer of the model descriptor is already configured",
    fixed = TRUE
  )
})

test_that("The optimizer is cloned during construction", {
  topt = t_opt("adam")

  obj = po("torch_optimizer", topt)
  expect_true(address(topt) != address(get_private(obj)$.optimizer))
})

test_that("PipeOpTorchOptimizer can be cloned", {
  obj1 = po("torch_optimizer")
  obj2 = obj1$clone(deep = TRUE)
  expect_deep_clone(obj1, obj2)
  # parameter set references are preserved
  expect_equal(address(obj2$param_set), address(get_private(obj2)$.optimizer$param_set))
})

test_that("phash works", {
  expect_equal(
    po("torch_optimizer", optimizer = "adam", lr = 1)$phash,
    po("torch_optimizer", optimizer = "adam", lr = 2)$phash
  )
  expect_false(
    po("torch_optimizer", "adam")$phash == po("torch_optimizer", "sgd")$phash
  )
})
