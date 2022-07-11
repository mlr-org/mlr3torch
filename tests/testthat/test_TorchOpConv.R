test_that("TorchOpConv1D", {
  task = tsk("iris")
  op = top("conv1d")

  for (i in seq_len(3)) {
    param_vals = list(
      out_channels = sample(1:3, 1),
      kernel_size = sample(1:5, 1),
      stride = sample(1:5, 1),
      padding = sample(1:5, 1),
      dilation = sample(1:5, 1),
      padding_mode = sample(c("zeros", "reflect", "replicate", "circular"), 1)
    )

    shape = c(sample(1:10, 1), sample(20:30, 2))

    inputs = list(input = invoke(torch_randn, .args = shape))

    expect_torchop(
      op = op,
      inputs = inputs,
      param_vals = param_vals,
      task = task
    )
  }
})

test_that("TorchOpConv2D", {
  task = tsk("iris")
  op = top("conv2d")

  for (i in seq_len(3)) {
    param_vals = list(
      out_channels = sample(1:3, 1),
      kernel_size = sample(1:5, 1),
      stride = sample(1:5, 1),
      padding = sample(1:5, 1),
      dilation = sample(1:5, 1),
      padding_mode = sample(c("zeros", "reflect", "replicate", "circular"), 1)
    )

    shape = c(sample(1:10, 1), sample(20:30, 3))

    inputs = list(input = invoke(torch_randn, .args = shape))

    expect_torchop(
      op = op,
      inputs = inputs,
      param_vals = param_vals,
      task = task
    )
  }
})

test_that("TorchOpConv3D", {
  task = tsk("iris")
  op = top("conv3d")

  for (i in seq_len(3)) {
    param_vals = list(
      out_channels = sample(1:3, 1),
      kernel_size = sample(1:5, 1),
      stride = sample(1:5, 1),
      padding = sample(1:5, 1),
      dilation = sample(1:5, 1),
      padding_mode = sample(c("zeros", "replicate", "circular"), 1)
    )

    shape = c(sample(1:10, 1), sample(20:30, 4))

    inputs = list(input = invoke(torch_randn, .args = shape))

    expect_torchop(
      op = op,
      inputs = inputs,
      param_vals = param_vals,
      task = task
    )
  }
})
