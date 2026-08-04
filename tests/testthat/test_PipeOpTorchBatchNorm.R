test_that("PipeOpTorchBatchNorm1D autotest", {
  po_test = po("nn_batch_norm1d")
  task = tsk("iris")
  graph1 = po("torch_ingress_num") %>>% po_test
  graph2 = po("torch_ingress_num") %>>% po("nn_unsqueeze", dim = 2) %>>% po_test

  expect_pipeop_torch(graph1, "nn_batch_norm1d", task)
  expect_pipeop_torch(graph2, "nn_batch_norm1d", task)
})

test_that("PipeOpTorchBatchNorm1D paramtest", {
  res = expect_paramset(po("nn_batch_norm1d"), nn_batch_norm1d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("PipeOpTorchBatchNorm2D autotest", {
  po_test = po("nn_batch_norm2d")
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_batch_norm2d", task)
})

test_that("PipeOpTorchBatchNorm2D paramtest", {
  res = expect_paramset(po("nn_batch_norm2d"), nn_batch_norm2d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("PipeOpTorchBatchNorm3D autotest", {
  po_test = po("nn_batch_norm3d")
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>%
    po("nn_reshape", shape = c(-1, 3, 64, 8, 8)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_batch_norm3d", task)
})

test_that("PipeOpTorchBatchNorm3D paramtest", {
  res = expect_paramset(po("nn_batch_norm3d"), nn_batch_norm3d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("jit_trace works (#354)", {
  graph = po("torch_ingress_num") %>>%
    nn("batch_norm1d") %>>%
    nn("head") %>>%
    po("torch_loss", t_loss("cross_entropy")) %>>%
    po("torch_optimizer", t_opt("adamw")) %>>%
    po("torch_model_classif", epochs = 1, batch_size = 50)
  lrn = as_learner(graph)
  task = tsk("iris")
  lrn$train(task)
  expect_prediction(lrn$predict(task))
})

test_that("shape inference matches the operator", {
  expect_shape_inference("nn_batch_norm1d", list(), c(2, 3, 17))
  expect_shape_inference("nn_batch_norm2d", list(), c(2, 3, 8, 8))
  expect_shape_inference("nn_batch_norm3d", list(), c(2, 3, 5, 5, 5))
})

test_that("shape inference requires the feature dimension", {
  expect_error(po("nn_batch_norm2d")$shapes_out(list(c(NA, NA, 17, 19))),
    "requires the feature dimension (dimension 2) of the input shape to be known", fixed = TRUE)
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  for (d in 1:3) {
    expect_shape_inference(sprintf("nn_batch_norm%id", d), generators = gen_shape(d + 2L))
  }
})
