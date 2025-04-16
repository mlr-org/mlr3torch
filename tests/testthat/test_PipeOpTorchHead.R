test_that("PipeOpTorchHead autotest", {
  po_test = po("nn_head")
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po_test

  expect_pipeop_torch(graph, "nn_head", task, "nn_linear")
})


test_that("PipeOpTorchHead paramtest", {
  po_test = po("nn_head")
  res = expect_paramset(po_test, torch::nn_linear, exclude = c("out_features", "in_features"))
  expect_paramtest(res)
})


test_that("correct error message", {
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po("nn_head")
  expect_error(graph$train(task), "expects 2D input")
})

test_that("correct output dim", {
  po_test = po("nn_head")
  graph = po("torch_ingress_num") %>>% po_test
  # binary
  task = tsk("iris")
  expect_equal(po_test$shapes_out(list(c(NA, 4)), task = task), list(output = c(NA, 3)))
  expect_equal( graph$train(task)[[1L]]$graph$pipeops$nn_head$module$weight$shape[1], 3)
  # multiclass
  task = tsk("sonar")
  expect_equal(po_test$shapes_out(list(c(NA, 60)), task = task), list(output = c(NA, 1)))
  expect_equal( graph$train(task)[[1L]]$graph$pipeops$nn_head$module$weight$shape[1], 1)
  # regression
  task = tsk("mtcars")
  expect_equal(po_test$shapes_out(list(c(NA, 11)), task = task), list(output = c(NA, 1)))
  expect_equal( graph$train(task)[[1L]]$graph$pipeops$nn_head$module$weight$shape[1], 1)
})
