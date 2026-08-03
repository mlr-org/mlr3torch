test_that("PipeOpTorchConv1 autotest", {
  po_conv = po("nn_conv1d", kernel_size = 2, out_channels = 3)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po("nn_unsqueeze", dim = 2) %>>% po_conv

  expect_pipeop_torch(graph, "nn_conv1d", task)
})

test_that("PipeOpTorchConv1d paramtest", {
  res = expect_paramset(po("nn_conv1d"), nn_conv1d, exclude = "in_channels")
  expect_paramtest(res)
})

test_that("PipeOpTorchConv2 autotest", {
  po_conv = po("nn_conv2d", kernel_size = 2, out_channels = 2)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_conv

  expect_pipeop_torch(graph, "nn_conv2d", task)
})

test_that("PipeOpTorchConv2d paramtest", {
  res = expect_paramset(po("nn_conv2d"), nn_conv2d, exclude = "in_channels")
  expect_paramtest(res)
})

test_that("PipeOpTorchConv3 autotest", {
  po_conv = po("nn_conv3d", kernel_size = 2, out_channels = 2)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>%
    po("nn_unsqueeze", dim = 5) %>>%
    po("nn_reshape", shape = c(-1, 3, 64, 8, 8)) %>>% po_conv

  expect_pipeop_torch(graph, "nn_conv3d", task)
})

test_that("PipeOpTorchConv3d paramtest", {
  res = expect_paramset(po("nn_conv3d"), nn_conv3d, exclude = "in_channels")
  expect_paramtest(res)
})


sampler_conv = function(dim, batch = TRUE) {
  list(
    conv_dim = dim,
    shape_in = sample(20:25, size = dim + 1 + as.integer(batch), replace = TRUE),
    out_channels = sample(1:3, size = 1, replace = TRUE),
    kernel_size = sample(5:6, size = dim, replace = TRUE),
    stride = sample(1:3, size = dim, replace = TRUE),
    padding = sample(1:2, size = dim, replace = TRUE),
    dilation = sample(1:2, size = dim, replace = TRUE),
    padding_mode = sample(c("zeros", "reflect", "replicate"), 1)
    # there is something wrong with circular padding: https://github.com/mlverse/torch/issues/940
  )
}

test_that("conv_output_shape works", {
  for (dim in 1:3) {
    testcase = sampler_conv(dim)
    mg = switch(dim,
      nn_conv1d,
      nn_conv2d,
      nn_conv3d,
    )
    args = testcase[names(testcase) %in% formalArgs(mg)]
    args$in_channels = testcase$shape_in[2L]
    m = do.call(mg, args = args)
    outshape = with_no_grad(m(do.call(torch_randn, args = list(unname(testcase$shape_in)))))$shape
    args1 = testcase[names(testcase) %in% formalArgs(conv_output_shape)]
    expect_true(all(outshape == do.call(conv_output_shape, args = args1)))
  }
})

test_that("conv_output_shape requires a batch dimension", {
  expect_error(
    conv_output_shape(shape_in = c(3, 20, 20), conv_dim = 2, padding = 1, dilation = 1, stride = 1,
      kernel_size = 3, out_channels = 2),
    "length 4"
  )
})

test_that("shape inference matches the operator", {
  expect_shapes_out_torch("nn_conv1d", list(out_channels = 5, kernel_size = 3), c(2, 3, 17))
  expect_shapes_out_torch("nn_conv2d", list(out_channels = 5, kernel_size = 3, stride = 2, padding = 1), c(2, 3, 17, 19))
  expect_shapes_out_torch("nn_conv2d", list(out_channels = 4, kernel_size = 3, dilation = 2), c(2, 3, 16, 16))
  expect_shapes_out_torch("nn_conv3d", list(out_channels = 5, kernel_size = 3), c(2, 3, 9, 9, 9))
  expect_equal(po("nn_conv2d", out_channels = 4, kernel_size = 3, padding_mode = "zeros")$
    shapes_out(list(c(2L, 3L, 8L, 8L)))[[1L]], c(2L, 4L, 6L, 6L))
})

test_that("shape inference requires the batch dimension, the channels and a non-empty output", {
  expect_error(po("nn_conv2d", out_channels = 4, kernel_size = 3)$shapes_out(list(c(3L, 17L, 19L))),
    "requires an input with 4 dimensions", fixed = TRUE)
  expect_error(po("nn_conv2d", out_channels = 5, kernel_size = 3)$shapes_out(list(c(NA, NA, 17, 19))),
    "requires the channel dimension (dimension 2) of the input shape to be known", fixed = TRUE)
  expect_error(po("nn_conv2d", out_channels = 4, kernel_size = 9)$shapes_out(list(c(2L, 3L, 4L, 4L))),
    "which no tensor can have", fixed = TRUE)
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  for (d in 1:3) {
    spec = list(rank = d + 2L, params = function() {
      list(out_channels = sample(1:4, 1L), kernel_size = sample(1:3, 1L), stride = sample(1:2, 1L),
        padding = sample(0:1, 1L), dilation = sample(1:2, 1L))
    })
    expect_shape_inference_sampled(sprintf("nn_conv%id", d), spec)
  }
})
