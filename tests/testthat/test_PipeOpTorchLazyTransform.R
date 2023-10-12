test_that("PipeOpTorchLazyTransform works for preprocessing", {
  task = nano_mnist()
  po_resize = po("transform_resize", size = c(10, 10))

  task1 = po_resize$train(list(task))[[1L]]

  lt = task1$data(cols = "image")

})

test_that("PipeOpTorchLazyTransform works in Graph", {
  task = nano_mnist()

  graph = po("torch_ingress_ltnsr") %>>%
    po("transform_resize", size = c(10, 10)) %>>%
    po("nn_flatten") %>>%
    po("nn_head") %>>%
    po("torch_optimizer") %>>%
    po("torch_loss", "cross_entropy") %>>%
    po("torch_model_classif", epochs = 1L, device = "cpu", batch_size = 16)

  graph$train(task)[[1L]]

})
