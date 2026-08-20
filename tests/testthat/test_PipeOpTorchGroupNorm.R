test_that("PipeOpTorchGroupNorm autotest", {
  po_test = po("nn_group_norm", num_groups = 2)
  task = tsk("iris")
  graph1 = po("torch_ingress_num") %>>% po_test
  graph2 = po("torch_ingress_num") %>>% po("nn_unsqueeze", dim = 3) %>>% po_test

  expect_pipeop_torch(graph1, "nn_group_norm", task)
  expect_pipeop_torch(graph2, "nn_group_norm", task)
})

test_that("PipeOpTorchGroupNorm paramtest", {
  res = expect_paramset(po("nn_group_norm"), nn_group_norm, exclude = "num_channels")
  expect_paramtest(res)
})

test_that("PipeOpTorchGroupNorm works on images", {
  po_test = po("nn_group_norm", num_groups = 3)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_group_norm", task)
})

test_that("shape inference matches the operator", {
  expect_shape_inference("nn_group_norm", list(num_groups = 2), c(2, 4))
  expect_shape_inference("nn_group_norm", list(num_groups = 2), c(2, 4, 6))
  expect_shape_inference("nn_group_norm", list(num_groups = 3), c(2, 3, 8, 8))
})

test_that("shape inference requires the feature dimension", {
  expect_error(po("nn_group_norm", num_groups = 2)$shapes_out(list(c(NA, NA, 17L))),
    "requires the feature dimension (dimension 2) of the input shape to be known", fixed = TRUE)
  expect_error(po("nn_group_norm", num_groups = 2)$shapes_out(list(c(NA))),
    "requires an input with at least 2 dimensions", fixed = TRUE)
})

test_that("shape inference requires 'num_groups' to divide the channels", {
  expect_error(po("nn_group_norm", num_groups = 3)$shapes_out(list(c(NA, 4L, 8L))),
    "requires 'num_groups' (3) to divide the number of channels", fixed = TRUE)
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  # the drawn shapes have an even feature dimension, so both group counts are always admissible
  for (rank in 2:4) {
    expect_shape_inference("nn_group_norm", params = function() list(num_groups = sample(c(1L, 2L), 1L)),
      generators = gen_shape(rank))
  }
})
