test_that("PipeOpTorchLazyTransform basic checks", {
  trafo = function(x) torchvision::transform_resize(x, c(10, 10))
  po_lt = po("trafo_lazy", trafo, packages = "R6")
  expect_pipeop(po_lt)
  expect_true("R6" %in% po_lt$packages)

  expect_error(
    po("trafo_lazy", function(x) torchvision::transform_resize(x, c(10, 10)), param_set = ps(augment = p_lgl()))
  )

  taskin = nano_mnist()

  taskout = po_lt$train(list(taskin))[[1L]]

  expect_permutation(taskin$feature_names, taskout$feature_names)

  # transformation is applied as we expect it
  expect_true(torch_equal(
    trafo(materialize(taskin$data(cols = "image")[[1L]])),
    materialize(taskout$data(cols = "image")[[1L]])
  ))

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

test_that("PipeOpTorchLazyTransform works with")
