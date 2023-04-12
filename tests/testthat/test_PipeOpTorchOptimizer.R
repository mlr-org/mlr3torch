test_that("PipeOpTorchOptimizer basic checks", {
  graph = po("torch_ingress_num") %>>%
    po("torch_optimizer", optimizer = t_opt("sgd"), lr = 0.123)

  task = tsk("iris")
  md = graph$train(task)[[1L]]

  expect_class(md$optimizer, "TorchOptimizer")
  expect_class(md$optimizer$generator, "optim_sgd")
  expect_true(md$optimizer$param_set$values$Lr == 0.123)

})
test_that("PipeOpTorchOptimizer can be cloned", {
  obj = po("torch_optimizer")
  addr1 = data.table::address(get_private(obj)$.optimizer$param_set)
  addr2 = data.table::address(obj$param_set)

  expect_identical(addr1, addr2)

  obj1 = obj$clone()
  addr11 = data.table::address(get_private(obj1)$.optimizer$param_set)
  addr12 = data.table::address(obj1$param_set)

  expect_identical(addr11, addr12)
})

