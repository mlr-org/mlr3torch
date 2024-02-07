test_that("PipeOpTorchReshape autotest", {
  obj = po("nn_reshape", shape = c(-1, 2, 2))
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% obj

  expect_pipeop_torch(graph, "nn_reshape", task)

  out = po("nn_reshape", shape = c(NA, 2, 2))$shapes_out(list(input = c(1, 4)))
  expect_true(!is.character(all.equal(out[[1L]], c(NA, 2, 2))))
})

test_that("PipeOpTorchReshape paramtest", {
  res = expect_paramset(po("nn_reshape"), nn_reshape)
  expect_paramtest(res)
})

test_that("PipeOpTorchUnsqueeze autotest", {
  obj = po("nn_unsqueeze", dim = 3)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% obj

  expect_pipeop_torch(graph, "nn_unsqueeze", task)
})

test_that("PipeOpTorchUnsqueeze paramtest", {
  res = expect_paramset(po("nn_unsqueeze"), nn_unsqueeze)
  expect_paramtest(res)
})

test_that("PipeOpTorchSqueeze autotest", {
  obj = po("nn_squeeze", dim = 3)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po("nn_unsqueeze", dim = 3) %>>%  obj

  x = po("nn_squeeze")

  expect_pipeop_torch(graph, "nn_squeeze", task)
})

test_that("PipeOpTorchSqueeze paramtest", {
  res = expect_paramset(po("nn_unsqueeze"), nn_unsqueeze)
  expect_paramtest(res)
})


test_that("PipeOpTorchFlatten autotest", {
  obj = po("nn_flatten", start_dim = 2, end_dim = 4)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% obj
  expect_pipeop_torch(graph, "nn_flatten", task)
})

test_that("PipeOpTorchFlatten", {
  res = expect_paramset(po("nn_flatten"), nn_flatten)
  expect_paramtest(res)
})
