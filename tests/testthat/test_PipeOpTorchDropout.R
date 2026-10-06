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
  expect_shape_inference("nn_dropout", list(p = 0.5), c(2, 4, 6))
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  expect_shape_inference("nn_dropout", list(p = 0.5), generators = gen_shape(3L))
})

test_that("PipeOpTorchDropout2D autotest", {
  po_test = po("nn_dropout2d")
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_dropout2d", task)
})

test_that("PipeOpTorchDropout2D paramtest", {
  res = expect_paramset(po("nn_dropout2d"), nn_dropout2d)
  expect_paramtest(res)
})

test_that("PipeOpTorchDropout3D autotest", {
  po_test = po("nn_dropout3d")
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>%
    po("nn_reshape", shape = c(-1, 3, 64, 8, 8)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_dropout3d", task)
})

test_that("PipeOpTorchDropout3D paramtest", {
  res = expect_paramset(po("nn_dropout3d"), nn_dropout3d)
  expect_paramtest(res)
})

test_that("the channel-wise dropouts require an input of the right rank", {
  # torch accepts a rank-3 input for `nn_dropout2d` but only warns that it is guessing which
  # dimension holds the channels
  expect_error(po("nn_dropout2d")$shapes_out(list(c(NA, 3L, 8L))),
    "requires an input with 4 dimensions", fixed = TRUE)
  expect_error(po("nn_dropout3d")$shapes_out(list(c(NA, 3L, 8L, 8L))),
    "requires an input with 5 dimensions", fixed = TRUE)
})

test_that("shape inference agrees with the module for the channel-wise dropouts", {
  expect_shape_inference("nn_dropout2d", list(p = 0.5), c(2, 3, 8, 8))
  expect_shape_inference("nn_dropout3d", list(p = 0.5), c(2, 3, 4, 4, 4))
  expect_shape_inference("nn_dropout2d", list(p = 0.5), generators = gen_shape(4L))
  expect_shape_inference("nn_dropout3d", list(p = 0.5), generators = gen_shape(5L))
})
