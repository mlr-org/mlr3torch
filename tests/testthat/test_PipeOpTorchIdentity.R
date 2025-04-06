test_that("PipeOpTorchIdentity works", {
  po_identity = po("nn_identity")
  graph = po("torch_ingress_num") %>>% po_identity
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_identity", task, "nn_identity")
})

test_that("PipeOpTorchIdentity paramtest", {
  po_identity = po("nn_identity")
  res = expect_paramset(po_identity, nn_identity)
  expect_paramtest(res)
})