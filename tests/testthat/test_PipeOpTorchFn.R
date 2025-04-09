test_that("PipeOpTorchFn autotest", {
  po_test = po("nn_fn", fn = function(tnsr) tnsr * 2)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po_test

  expect_pipeop_torch(graph, "nn_fn", task)
})

test_that("PipeOpTorchFn works for a simple function", {
  withr::local_options(mlr3torch.cache = TRUE)

  # for the nano imagenet data, gets the blue channel
  extract_blue_channel = function(x) x[, 3, , ]
  po = po("nn_fn", fn = extract_blue_channel)
  graph = po("torch_ingress_ltnsr") %>>% po

  task = nano_imagenet()
  task_dt = task$data()

  # create a batch of size 1
  tnsr = materialize(task_dt$image[1])[[1]]$unsqueeze(dim = 1)
  blue_channel = extract_blue_channel(tnsr)

  md_trained = graph$train(task)[[1]]
  trained = md_trained$graph$train(tnsr)[[1]]

  expect_true(torch_equal(blue_channel, trained))
})

test_that("PipeOpTorchFn works for a function with extra arguments", {
  withr::local_options(mlr3torch.cache = TRUE)

  # for the nano imagenet data, gets the blue channel
  extract_channel = function(x, channel_idx) x[, channel_idx, , ]
  po = po("nn_fn", fn = extract_channel, channel_idx = 3)
  graph = po("torch_ingress_ltnsr") %>>% po

  task = nano_imagenet()
  task_dt = task$data()

  # create a batch of size 1
  tnsr = materialize(task_dt$image[1])[[1]]$unsqueeze(dim = 1)
  blue_channel = extract_channel(tnsr, 3)

  md_trained = graph$train(task)[[1]]
  trained = md_trained$graph$train(tnsr)[[1]]

  expect_true(torch_equal(blue_channel, trained))
})

test_that("PipeOpTorchFn works with a user-provided shapes_out fn", {
  withr::local_options(mlr3torch.cache = TRUE)
  extract_channel = function(x, channel_idx) x[, channel_idx, , ]
  so_extract_channel = function(shapes_in, param_vals, task) {
    sin = shapes_in[[1L]]
    batch_dim = sin[1L]
    batchdim_is_unknown = is.na(batch_dim)
    if (batchdim_is_unknown) {
      sin[1] = 1L
    }
    sout_dims = sin[-2]
    if (batchdim_is_unknown) {
      sout_dims[1] = NA
    }
    return(setNames(list(sout_dims), "output"))
  }

  po = po("nn_fn", fn = extract_channel, channel_idx = 3, shapes_out = so_extract_channel)
  graph = po("torch_ingress_ltnsr") %>>% po

  task = nano_imagenet()
  task_dt = task$data()

  tnsr = materialize(task_dt$image[1])[[1]]$unsqueeze(dim = 1)
  blue_channel = extract_channel(tnsr, 3)

  md_trained = graph$train(task)[[1]]
  trained = md_trained$graph$train(tnsr)[[1]]

  expect_true(torch_equal(blue_channel, trained))
})
