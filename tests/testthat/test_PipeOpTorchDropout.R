test_that("PipeOpTorchDropout autotest", {
  po_test = po("nn_dropout")
  graph = po("torch_ingress_num") %>>% po_test

  expect_pipeop_torch(graph, "nn_dropout", tsk("iris"))
})


test_that("PipeOpTorchDropout paramtest", {
  res = expect_paramset(po("nn_dropout"), nn_dropout)
  expect_paramtest(res)
})
