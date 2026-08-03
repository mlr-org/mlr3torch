test_that("Basic properties", {
  expect_pipeop(po("nn_conv1d"))
  expect_pipeop(po("nn_conv2d"))
  expect_pipeop(po("nn_conv3d"))
})


test_that("PipeOpTorchConvTranspose1D autotest", {
  po_conv = po("nn_conv_transpose1d", kernel_size = 2, out_channels = 4)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po("nn_unsqueeze", dim = 2) %>>% po_conv

  expect_pipeop(po_conv)


  expect_pipeop_torch(graph, "nn_conv_transpose1d", task)
})

test_that("PipeOpTorchConvTranspose1D paramtest", {
  res = expect_paramset(po("nn_conv_transpose1d"), nn_conv_transpose1d, exclude = "in_channels")
  expect_paramtest(res)
})

test_that("PipeOpTorchConvTranspose2D autotest", {
  po_conv = po("nn_conv_transpose2d", kernel_size = 2, out_channels = 4)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_conv

  expect_pipeop_torch(graph, "nn_conv_transpose2d", task)
})

test_that("PipeOpTorchConvTranspose2D paramtest", {
  res = expect_paramset(po("nn_conv_transpose2d"), nn_conv_transpose2d, exclude = "in_channels")
  expect_paramtest(res)
})

test_that("PipeOpTorchConvTranspose3D autotest", {
  po_conv = po("nn_conv_transpose3d", kernel_size = 2, out_channels = 4)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>%
    po("nn_reshape", shape = c(-1, 3, 64, 8, 8)) %>>%
    po_conv

  expect_pipeop_torch(graph, "nn_conv_transpose3d", task)
})

test_that("PipeOpTorchConvTranspose3D paramtest", {
  res = expect_paramset(po("nn_conv_transpose3d"), nn_conv_transpose3d, exclude = "in_channels")
  expect_paramtest(res)
})

sampler_conv_transpose = function(dim) {
  stride = sample(1:3, size = dim, replace = TRUE)
  list(
    dim = dim,
    shape_in = sample(20:25, size = dim + 2L, replace = TRUE),
    out_channels = sample(1:3, size = 1, replace = TRUE),
    kernel_size = sample(5:6, size = dim, replace = TRUE),
    stride = stride,
    padding = sample(1:2, size = dim, replace = TRUE),
    dilation = sample(1:2, size = dim, replace = TRUE),
    # torch requires the output padding to be smaller than the stride or the dilation
    output_padding = sample.int(min(stride), 1L) - 1L
    # transposed convolutions only support the "zeros" padding mode, which is the default
  )
}

test_that("conv_transpose_output_shape works", {
  for (dim in 1:3) {
    testcase = sampler_conv_transpose(dim)
    mg = switch(dim,
      nn_conv_transpose1d,
      nn_conv_transpose2d,
      nn_conv_transpose3d,
    )
    args = testcase[names(testcase) %in% formalArgs(mg)]
    args$in_channels = testcase$shape_in[2L]
    m = do.call(mg, args = args)
    in_tensor = do.call(torch_randn, args = list(unname(testcase$shape_in)))
    outshape = with_no_grad(m(in_tensor))$shape
    args1 = testcase[names(testcase) %in% formalArgs(conv_transpose_output_shape)]
    expect_true(all(outshape == do.call(conv_transpose_output_shape, args = args1)))
  }
})

test_that("conv_transpose_output_shape requires a batch dimension", {
  # the PipeOp rejects such a shape via assert_ndim() before the shape function is reached
  expect_error(
    conv_transpose_output_shape(shape_in = c(3, 20, 20), dim = 2, padding = 1, dilation = 1,
      stride = 1, kernel_size = 3, output_padding = 0, out_channels = 2),
    "length 4"
  )
})

test_that("shape inference matches the operator", {
  expect_shapes_out_torch("nn_conv_transpose1d", list(out_channels = 5, kernel_size = 3), c(2, 3, 17))
  expect_shapes_out_torch("nn_conv_transpose2d", list(out_channels = 5, kernel_size = 3, stride = 2), c(2, 3, 8, 9))
  expect_shapes_out_torch("nn_conv_transpose2d", list(out_channels = 5, kernel_size = 3), c(2, 3, 17, 19))
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  for (d in 1:3) {
    spec = list(rank = d + 2L, params = function() {
      list(out_channels = sample(1:4, 1L), kernel_size = sample(1:3, 1L), stride = sample(1:2, 1L),
        padding = sample(0:1, 1L))
    })
    expect_shape_inference_sampled(sprintf("nn_conv_transpose%id", d), spec)
  }
})
