test_that("PipeOpTorchLoad works", {
  graph = po("torch_ingress_num") %>>%
    po("torch_load")
})
