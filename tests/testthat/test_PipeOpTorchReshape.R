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

test_that("nn_unsqueeze interprets negative dim like torch", {
  x = torch_randn(3L, 4L, 6L)
  for (d in c(-1L, -2L, -3L, -4L)) {
    inferred = po("nn_unsqueeze", dim = d)$shapes_out(list(c(NA, 4L, 6L)))[[1L]]
    actual = dim(x$unsqueeze(d))
    expect_equal(length(inferred), length(actual), info = as.character(d))
    # the batch dimension is NA, compare the remaining ones
    expect_equal(inferred[!is.na(inferred)], actual[!is.na(inferred)], info = as.character(d))
  }
  expect_error(po("nn_unsqueeze", dim = 9L)$shapes_out(list(c(NA, 4L, 6L))))
})

test_that("nn_squeeze without dim squeezes all non-batch dimensions", {
  # `nn_squeeze()` had no default for `dim`, so the documented `dim = NULL` behaviour could be
  # inferred but never trained
  expect_equal(po("nn_squeeze")$shapes_out(list(c(NA, 1L, 4L, 1L)))[[1L]], c(NA, 4L))

  net = nn_squeeze()
  expect_equal(dim(net(torch_randn(3L, 1L, 4L, 1L))), c(3L, 4L))
  # the batch dimension is kept even when it is 1, so that the output matches the inferred shape
  expect_equal(dim(net(torch_randn(1L, 1L, 4L, 1L))), c(1L, 4L))
  expect_equal(dim(nn_squeeze(dim = 2L)(torch_randn(3L, 1L, 4L))), c(3L, 4L))
})
