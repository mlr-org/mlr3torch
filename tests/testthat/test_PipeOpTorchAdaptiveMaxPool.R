test_that("PipeOpTorchAdaptiveMaxPool1D works", {
  po_test = po("nn_adaptive_max_pool1d", output_size = 10)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po_test
  expect_pipeop_torch(graph, "nn_adaptive_max_pool1d", task)
})

test_that("PipeOpTorchAdaptiveMaxPool1D paramtest", {
  # return_indices is a construction argument.
  res = expect_paramset(po("nn_adaptive_max_pool1d"), nn_adaptive_max_pool1d, exclude = "return_indices")
  expect_paramtest(res)
})

test_that("PipeOpTorchAdaptiveMaxPool2D works with a 1d output size", {
  po_test = po("nn_adaptive_max_pool2d", output_size = 10)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_adaptive_max_pool2d", task)
})

test_that("PipeOpTorchAdaptiveMaxPool2D works with a 2d output size", {
  po_test = po("nn_adaptive_max_pool2d", output_size = c(8, 12))
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_adaptive_max_pool2d", task)
})

test_that("PipeOpTorchAdaptiveMaxPool2D paramtest", {
  # return_indices is a construction argument.
  res = expect_paramset(po("nn_adaptive_max_pool2d"), nn_adaptive_max_pool2d, exclude = "return_indices")
  expect_paramtest(res)
})

test_that("PipeOpTorchAdaptiveMaxPool3D works", {
  po_test = po("nn_adaptive_max_pool3d", output_size = c(10, 11, 12))
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>%
    po("nn_reshape", shape = c(-1, 3, 64, 8, 8)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_adaptive_max_pool3d", task)
})

test_that("PipeOpTorchAdaptiveMaxPool3D paramtest", {
  # return_indices is a construction argument.
  res = expect_paramset(po("nn_adaptive_max_pool3d"), nn_adaptive_max_pool3d, exclude = "return_indices")
  expect_paramtest(res)
})

test_that("return_indices adds a second output channel", {
  obj = po("nn_adaptive_max_pool2d", return_indices = TRUE, output_size = 4)
  expect_equal(obj$output$name, c("output", "indices"))
  # the indices say where each maximum came from, so they have the shape of the pooled output
  expect_equal(obj$shapes_out(list(c(NA, 3L, 16L, 16L))),
    list(output = c(NA, 3L, 4L, 4L), indices = c(NA, 3L, 4L, 4L)))
  # two operators that differ only in `return_indices` are not the same operator
  expect_false(obj$phash == po("nn_adaptive_max_pool2d", output_size = 4)$phash)
})

test_that("shape inference matches the operator", {
  expect_shape_inference("nn_adaptive_max_pool1d", list(output_size = 4), c(2, 3, 17))
  expect_shape_inference("nn_adaptive_max_pool2d", list(output_size = c(2, 3)), c(2, 3, 16, 20))
  expect_shape_inference("nn_adaptive_max_pool3d", list(output_size = c(2, 3, 4)), c(2, 3, 5, 7, 9))
  expect_shape_inference("nn_adaptive_max_pool2d", list(output_size = 2, return_indices = TRUE),
    c(2, 3, 16, 20))
})

test_that("shape inference requires the batch dimension and a non-empty output", {
  expect_error(po("nn_adaptive_max_pool2d", output_size = 4)$shapes_out(list(c(NA, 28L, 28L))),
    "requires an input with 4 dimensions", fixed = TRUE)
  expect_error(po("nn_adaptive_max_pool1d", output_size = 0)$shapes_out(list(c(2L, 3L, 8L))),
    "which no tensor can have", fixed = TRUE)
})

test_that("an unknown input extent still gives a known output", {
  # the output size is fixed by `output_size`, so rejecting an unknown input extent would throw
  # away information
  for (d in 1:2) {
    obj = po(sprintf("nn_adaptive_max_pool%id", d), output_size = rep(4L, d))
    shape_in = as.integer(c(NA, 3L, rep(NA_integer_, d)))
    expect_equal(obj$shapes_out(list(input = shape_in))[[1L]], c(NA, 3L, rep(4L, d)))
  }
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  for (d in 1:3) {
    expect_shape_inference(sprintf("nn_adaptive_max_pool%id", d),
      params = function() list(output_size = sample(1:4, d)),
      generators = gen_shape(d + 2L))
  }
})
