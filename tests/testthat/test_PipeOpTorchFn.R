test_that("PipeOpTorchFn works for a simple function", {
  withr::local_options(mlr3torch.cache = TRUE)
  
  # for the nano imagenet data, gets the blue channel
  drop_dim =  function(x) x[-1, , ]
  po = po("nn_fn", param_vals = list(fn = drop_dim))
  graph = po("torch_ingress_ltnsr") %>>% po

  task = nano_imagenet()
  task_dt = task$data()

  tnsr = materialize(task_dt$image[1])[[1]]
  blue_channel = drop_dim(tnsr)
  
  md_trained = graph$train(task)[[1]]
  trained = md_trained$graph$train(tnsr)[[1]]

  expect_true(torch_equal(blue_channel, trained))
})

test_that("PipeOpTorchFn works with a user-provided shapes_out fn", {
  withr::local_options(mlr3torch.cache = TRUE)
  
  drop_dim =  function(x) x[-1, , ]
  so_drop_dim = function(shapes_in, param_vals, task) {
    setNames(list(shapes_in[[1]]), "output")
  }
  po = po("nn_fn", param_vals = list(fn = drop_dim, shapes_out = so_drop_dim))
  graph = po("torch_ingress_ltnsr") %>>% po

  task = nano_imagenet()
  task_dt = task$data()

  tnsr = materialize(task_dt$image[1])[[1]]
  blue_channel = drop_dim(tnsr)
  
  md_trained = graph$train(task)[[1]]
  trained = md_trained$graph$train(tnsr)[[1]]

  expect_true(torch_equal(blue_channel, trained))
})
