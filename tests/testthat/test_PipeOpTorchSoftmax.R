test_that("PipeOpTorchSoftmin autotest", {
  po_test = po("nn_softmin", dim = 2)
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_softmin", task)
})

test_that("PipeOpTorchSoftmin paramtest", {
  res = expect_paramset(po("nn_softmin"), nn_softmin)
  expect_paramtest(res)
})

test_that("PipeOpTorchLogSoftmax autotest", {
  po_test = po("nn_log_softmax", dim = 2)
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_log_softmax", task)
})

test_that("PipeOpTorchLogSoftmax paramtest", {
  res = expect_paramset(po("nn_log_softmax"), nn_log_softmax)
  expect_paramtest(res)
})

test_that("PipeOpTorchSoftmax2D autotest", {
  po_test = po("nn_softmax2d")
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_softmax2d", task)
})

test_that("PipeOpTorchSoftmax2D paramtest", {
  res = expect_paramset(po("nn_softmax2d"), nn_softmax2d)
  expect_paramtest(res)
})

test_that("shape inference matches the operator", {
  expect_shape_inference("nn_softmin", list(dim = 2), c(2, 4, 6))
  expect_shape_inference("nn_log_softmax", list(dim = 2), c(2, 4, 6))
  expect_shape_inference("nn_softmax2d", list(), c(2, 3, 4, 4))
})

test_that("shape inference rejects a 'dim' that does not address a dimension", {
  expect_error(po("nn_softmin", dim = 9L)$shapes_out(list(c(2L, 4L))), "cannot use 'dim' 9",
    fixed = TRUE)
  expect_error(po("nn_log_softmax", dim = 9L)$shapes_out(list(c(2L, 4L))), "cannot use 'dim' 9",
    fixed = TRUE)
  # negative values are legal in torch and must be accepted
  expect_equal(po("nn_softmin", dim = -1)$shapes_out(list(c(2L, 4L)))[[1L]], c(2L, 4L))
  expect_equal(po("nn_log_softmax", dim = -1)$shapes_out(list(c(2L, 4L)))[[1L]], c(2L, 4L))
})

test_that("PipeOpTorchSoftmax2D requires a batched image", {
  # torch reads a rank-3 input as an unbatched (channel, height, width) tensor and would normalize
  # over the wrong dimension
  expect_error(po("nn_softmax2d")$shapes_out(list(c(NA, 3L, 8L))),
    "requires an input with 4 dimensions", fixed = TRUE)
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  expect_shape_inference("nn_softmin", params = function() list(dim = sample(c(2:3, -1L), 1L)),
    generators = gen_shape(3L))
  expect_shape_inference("nn_log_softmax", params = function() list(dim = sample(c(2:3, -1L), 1L)),
    generators = gen_shape(3L))
  expect_shape_inference("nn_softmax2d", generators = gen_shape(4L))
})
