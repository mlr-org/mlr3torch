test_that("Basic checks", {
  descriptor = DescriptorTorchOptimizer$new(
    torch_optimizer = optim_adam,
    label = "Adam",
    packages = "mypackage"
  )
  expect_equal(descriptor$id, "optim_adam")
  expect_r6(descriptor, "DescriptorTorchOptimizer")
  expect_set_equal(descriptor$packages, c("mypackage", "torch", "mlr3torch"))
  expect_equal(descriptor$label, "Adam")
  expect_set_equal(descriptor$param_set$ids(), setdiff(formalArgs(optim_adam), "params"))
  expect_error(descriptor$generate(), regexp = "could not be loaded: mypackage", fixed = TRUE)

  expect_error(
    DescriptorTorchOptimizer$new(
      torch_optimizer = optim_sgd,
      param_set = ps(lr = p_uty(), params = p_uty())
    ),
    regexp = "The name 'params' is reserved for the network parameters.", fixed = TRUE
  )

  expect_error(DescriptorTorchOptimizer$new(optim_adam, id = "mse", param_set = ps(par = p_uty())),
    regexp = "Parameter values with ids 'par' are missing in generator.", fixed = TRUE
  )

  descriptor1 = DescriptorTorchOptimizer$new(
    torch_optimizer = optim_sgd,
    label = "Stochastic Gradient Descent",
    id = "Sgd"
  )

  descriptor1$param_set$set_values(lr = 0.9191)
  expect_set_equal(descriptor1$packages, c("torch", "mlr3torch"))
  expect_equal(descriptor1$label, "Stochastic Gradient Descent")
  expect_equal(descriptor1$id, "Sgd")
  expect_equal(descriptor1$param_set$values$lr, 0.9191)

  opt = descriptor1$generate(nn_linear(1, 1)$parameters)
  expect_class(opt, "torch_optimizer")
  expect_equal(opt$defaults$lr, 0.9191)

  descriptor2 = DescriptorTorchOptimizer$new(
    torch_optimizer = optim_sgd,
    param_set = ps(lr = p_uty())
  )
  expect_equal(descriptor2$param_set$ids(), "lr")
})


test_that("dictionary retrieval works", {
  descriptor = t_opt("adam", lr = 0.99)
  expect_r6(descriptor, "DescriptorTorchOptimizer")
  expect_class(descriptor$generator, "optim_adam")
  expect_equal(descriptor$param_set$values$lr, 0.99)

  descriptors = t_opts(c("adam", "sgd"))
  expect_list(descriptors, types = "DescriptorTorchOptimizer")
  expect_identical(ids(descriptors), c("adam", "sgd"))

  expect_class(t_opt(), "DictionaryMlr3torchOptimizers")
  expect_class(t_opts(), "DictionaryMlr3torchOptimizers")
})


test_that("dictionary can be converted to a table", {
  tbl = as.data.table(mlr3torch_optimizers)
  expect_data_table(tbl, ncols = 3, key = "key")
  expect_equal(colnames(tbl), c("key", "label", "packages"))

})

test_that("Cloning works", {
  descriptor1 = t_opt("adam")
  descriptor2 = descriptor1$clone(deep = TRUE)
  expect_deep_clone(descriptor1, descriptor2)
})

test_that("Printer works", {
  observed = capture.output(print(t_opt("adam")))
  expected = c(
   "<DescriptorTorchOptimizer:adam> Adaptive Moment Estimation",
   "* Generator: optim_adam",
   "* Parameters: list()",
   "* Packages: torch,mlr3torch"
  )
  expect_identical(observed, expected)
})


test_that("Converters are correctly implemented", {
  expect_r6(as_descriptor_torch_optimizer("adam"), "DescriptorTorchOptimizer")
  descriptor = as_descriptor_torch_optimizer(optim_adam)
  expect_r6(descriptor, "DescriptorTorchOptimizer")
  expect_equal(descriptor$id, "optim_adam")
  expect_equal(descriptor$label, "Optim_adam")

  descriptor1 = as_descriptor_torch_optimizer(descriptor, clone = TRUE)
  expect_deep_clone(descriptor, descriptor1)

  descriptor2 = as_descriptor_torch_optimizer(optim_adam, id = "myopt", label = "Custom",
    man = "my_opt", param_set = ps(lr = p_uty())
  )
  expect_r6(descriptor2, "DescriptorTorchOptimizer")
  expect_equal(descriptor2$id, "myopt")
  expect_equal(descriptor2$label, "Custom")
  expect_equal(descriptor2$man, "my_opt")
  expect_equal(descriptor2$param_set$ids(), "lr")
})


test_that("Parameter test: adam", {
  descriptor = t_opt("adam")
  param_set = descriptor$param_set
  fn = descriptor$generator
  res = autotest_paramset(param_set, fn, exclude = "params")
  expect_paramtest(res)
})

test_that("Parameter test: sgd", {
  descriptor = t_opt("sgd")
  param_set = descriptor$param_set
  # lr is set to `optim_required()`
  fn = descriptor$generator
  res = autotest_paramset(param_set, fn, exclude = c("params", "lr"))
  expect_paramtest(res)
})

test_that("Parameter test: asgd", {
  descriptor = t_opt("asgd")
  param_set = descriptor$param_set
  fn = descriptor$generator
  res = autotest_paramset(param_set, fn, exclude = "params")
  expect_paramtest(res)
})

test_that("Parameter test: rprop", {
  descriptor = t_opt("rprop")
  param_set = descriptor$param_set
  fn = descriptor$generator
  res = autotest_paramset(param_set, fn, exclude = "params")
  expect_paramtest(res)
})

test_that("Parameter test: rmsprop", {
  descriptor = t_opt("rmsprop")
  param_set = descriptor$param_set
  fn = descriptor$generator
  res = autotest_paramset(param_set, fn, exclude = "params")
  expect_paramtest(res)
})

test_that("Parameter test: adagrad", {
  descriptor = t_opt("adagrad")
  param_set = descriptor$param_set
  fn = descriptor$generator
  res = autotest_paramset(param_set, fn, exclude = "params")
  expect_paramtest(res)
})

test_that("Parameter test: adadelta", {
  descriptor = t_opt("adadelta")
  param_set = descriptor$param_set
  fn = descriptor$generator
  res = autotest_paramset(param_set, fn, exclude = "params")
  expect_paramtest(res)
})


test_that("phash works", {
  expect_equal(t_opt("adam", lr = 2)$phash, t_opt("adam", lr = 1)$phash)
  expect_false(t_opt("sgd")$phash == t_opt("adam")$phash)
  expect_false(t_opt("sgd", id = "a")$phash == t_opt("adam", id = "b")$phash)
  expect_false(t_opt("sgd", label = "a")$phash == t_opt("adam", label = "b")$phash)
})
