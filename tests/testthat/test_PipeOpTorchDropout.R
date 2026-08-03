test_that("PipeOpTorchDropout autotest", {
  po_test = po("nn_dropout")
  graph = po("torch_ingress_num") %>>% po_test

  expect_pipeop_torch(graph, "nn_dropout", tsk("iris"))
})


test_that("PipeOpTorchDropout paramtest", {
  res = expect_paramset(po("nn_dropout"), nn_dropout)
  expect_paramtest(res)
})

test_that("shape inference matches the operator", {
  expect_shapes_out_torch("nn_dropout", list(p = 0.5), c(2, 4, 6))
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  expect_shape_inference_sampled("nn_dropout", list(rank = 3L, params = function() list(p = 0.5)))
})
