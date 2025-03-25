test_that("PipeOpTorchFn autotest", {
  withr::local_options(mlr3torch.cache = TRUE)
  po = po("nn_fn", fn = function(x, ...) x)
  graph = po("torch_ingress_num") %>>% po
  expect_pipeop_torch(graph, "nn_fn", tsk("iris"), "nn_fn")
})

test_that("PipeOpTorchFn drop dimensions", {
  withr::local_options(mlr3torch.cache = TRUE)
  
  # for the tiny imagenet data, should get only the blue channel
  po = po("nn_fn", fn = function(x) x[, -1])
  graph = po("torch_ingress_ltnsr") %>>% po

  task = tsk("tiny_imagenet")
  task_dt = task$data()
  
  graph$train(task)
  result = graph$predict(task)[[1]]
  result_dt = result$data()

  result_dt
})
