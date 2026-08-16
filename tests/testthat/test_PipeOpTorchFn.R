test_that("PipeOpTorchFn autotest", {
  po_test = po("nn_fn", fn = function(tnsr) tnsr * 2)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po_test

  expect_pipeop_torch(graph, "nn_fn", task)
})

test_that("PipeOpTorchFn works for a simple function", {
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

test_that("PipeOpTorchFn works for a user-provided ParamSet", {
  # for the nano imagenet data, gets the blue channel
  extract_channel = function(x, channel_idx) x[, channel_idx, , ]
  po = po("nn_fn", fn = extract_channel, param_set = ps(channel_idx = p_int(tags = "required")), channel_idx = 3)
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


test_that("shape inference falls back to tracing the function", {
  # `infer_shapes()` fills in different values for the unknown dimensions and marks those that
  # vary as unknown again
  obj = po("nn_fn", fn = function(x) x * 2)
  expect_equal(obj$shapes_out(list(c(NA, NA, 16L)))[[1L]], c(NA, NA, 16L))
})

test_that("phash takes the fn and shapes_out into account", {
  # `hash_input()`'s deparse of the body drops the names of the arguments, so these two used to hash
  # equal even though they select different columns
  f1 = function(x, a) torch_narrow(x, 2, start = 1, length = a)
  f2 = function(x, a) torch_narrow(x, 2, length = 1, start = a)
  expect_false(po("nn_fn", fn = f1)$phash == po("nn_fn", fn = f2)$phash)

  # two closures crated from one definition share their body and differ only in what they capture
  mk = function(k) po("nn_fn", fn = mlr3misc::crate(function(x) x * k, k))
  expect_false(mk(2)$phash == mk(3)$phash)

  fn = function(x, a) x + a
  expect_equal(po("nn_fn", fn = fn)$phash, po("nn_fn", fn = fn)$phash)
  expect_equal(po("nn_fn", fn = fn)$phash, po("nn_fn", fn = fn)$clone(deep = TRUE)$phash)

  # `shapes_out` is part of the configuration as well
  shapes_out = function(shapes_in, param_vals, task) shapes_in
  expect_false(po("nn_fn", fn = fn, shapes_out = shapes_out)$phash == po("nn_fn", fn = fn)$phash)
})
