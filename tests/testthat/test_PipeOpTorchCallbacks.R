test_that("PipeOpTorchCallbacks works", {
  obj = po("torch_callbacks")
  expect_class(obj, "PipeOpTorchCallbacks")

  md = po("torch_ingress_num")$train(list(tsk("iris")))[[1L]]

  result = obj$train(list(md))[[1L]]



})
