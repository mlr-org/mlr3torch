test_that("PipeOpTorchMaxPool1D works", {
  po_test = po("nn_max_pool1d", kernel_size = 2)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po_test
  expect_pipeop_torch(graph, "nn_max_pool1d", task)
})

test_that("PipeOpTorchMaxPool1D paramtest", {
  # return_indices is a construction argument.
  res = expect_paramset(po("nn_max_pool1d"), nn_max_pool1d, exclude = "return_indices")
  expect_paramtest(res)
})

test_that("PipeOpTorchMaxPool2D autotest", {
  po_test = po("nn_max_pool2d", kernel_size = 3)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_max_pool2d", task)
})

test_that("PipeOpTorchMaxPool2D paramtest", {
  # return_indices is a construction argument.
  res = expect_paramset(po("nn_max_pool2d"), nn_max_pool2d, exclude = "return_indices")
  expect_paramtest(res)
})

test_that("PipeOpTorchMaxPool3D autotest", {
  po_test = po("nn_max_pool3d", kernel_size = c(2, 3, 4))
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>%
    po("nn_reshape", shape = c(-1, 3, 64, 8, 8)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_max_pool3d", task)
})

test_that("PipeOpTorchMaxPool3D paramtest", {
  # return_indices is a construction argument.
  res = expect_paramset(po("nn_max_pool3d"), nn_max_pool3d, exclude = "return_indices")
  expect_paramtest(res)
})

sampler_max_pool = function(dim) {
  list(
    shape_in = sample(20:25, size = dim + 2L, replace = TRUE),
    conv_dim = dim,
    padding = sample(1:2, size = dim, replace = TRUE),
    stride = sample(1:3, size = dim, replace = TRUE),
    kernel_size = sample(5:6, size = dim, replace = TRUE),
    ceil_mode = sample(c(TRUE, FALSE), 1)
  )
}

test_that("max_output_shape works", {
  for (dim in 1:3) {
    testcase = sampler_max_pool(dim)
    mg = switch(dim,
      nn_max_pool1d,
      nn_max_pool2d,
      nn_max_pool3d
    )
    m = do.call(mg, testcase[names(testcase) %in% formalArgs(mg)])
    outshape = with_no_grad(m(do.call(torch::torch_randn, args = list(unname(testcase$shape_in)))))$shape
    expect_true(all(outshape == do.call(max_output_shape, args = testcase)))
  }
})

test_that("max_output_shape requires a batch dimension", {
  expect_error(
    max_output_shape(shape_in = c(3, 20, 20), conv_dim = 2, padding = 1, stride = 1,
      kernel_size = 3),
    "length 4"
  )
})

test_that("shape inference matches the operator", {
  expect_shapes_out_torch("nn_max_pool1d", list(kernel_size = 2), c(2, 3, 16))
  expect_shapes_out_torch("nn_max_pool1d", list(kernel_size = 2, stride = 2, padding = 1, ceil_mode = TRUE), c(2, 2, 5))
  expect_shapes_out_torch("nn_max_pool2d", list(kernel_size = 2), c(2, 3, 16, 16))
  expect_shapes_out_torch("nn_max_pool2d", list(kernel_size = 2, stride = 1, dilation = 2), c(2, 2, 8, 8))
  expect_shapes_out_torch("nn_max_pool2d", list(kernel_size = 3, stride = 2, dilation = 3, padding = 1), c(2, 2, 16, 20))
  expect_shapes_out_torch("nn_max_pool2d", list(kernel_size = 2, stride = 3, ceil_mode = TRUE), c(2, 2, 6, 6))
  # torch supports a per-dimension dilation for max pooling
  expect_equal(po("nn_max_pool2d", kernel_size = 3, dilation = c(1, 2))$
    shapes_out(list(c(2L, 3L, 8L, 8L)))[[1L]], c(2L, 3L, 2L, 2L))
})

test_that("shape inference requires the batch dimension and a non-empty output", {
  expect_error(po("nn_max_pool2d", kernel_size = 2)$shapes_out(list(c(NA, 28L, 28L))),
    "requires an input with 4 dimensions", fixed = TRUE)
  expect_error(po("nn_max_pool2d", kernel_size = 20, stride = 1)$shapes_out(list(c(NA, 3L, 8L, 8L))),
    "which no tensor can have", fixed = TRUE)
  expect_error(po("nn_max_pool1d", kernel_size = 0)$shapes_out(list(c(2L, 3L, 8L))),
    "which no tensor can have", fixed = TRUE)
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  for (d in 1:3) {
    # unlike average pooling, max pooling has a dilation
    spec = list(rank = d + 2L, params = function() {
      list(kernel_size = sample(1:3, 1L), stride = sample(1:2, 1L), padding = 0L,
        ceil_mode = sample(c(TRUE, FALSE), 1L), dilation = sample(1:2, 1L))
    })
    expect_shape_inference_sampled(sprintf("nn_max_pool%id", d), spec)
  }
})
