test_that("PipeOpTorchFn autotest", {
  po = po("nn_fn", fn = function(x, ...) x)
  # browser()
  graph = po("torch_ingress_num") %>>% po
  expect_pipeop_torch(graph, "nn_fn", tsk("iris"), "nn_fn")
})
