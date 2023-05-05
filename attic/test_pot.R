test_that("pot works", {
  # for nn_
  obj = pot("linear")
  expect_r6(obj, "PipeOpTorchLinear")
  expect_true(obj$id == "nn_linear")
  # for torch_optimizer
  obj = pot("optimizer", "adam")
  expect_r6(obj, "PipeOpTorchOptimizer")
  expect_true(obj$id == "torch_optimizer")
})

test_that("pot gives informative error message", {
  expect_error(pot("nn_linear"), regexp = "You probably wanted po(\"nn_", fixed = TRUE)
  expect_error(pot("torch_optimizer"), regexp = "You probably wanted po(\"torch_", fixed = TRUE)
  expect_error(pot("xyz"), regexp = "PipeOp with neither id torch_xyz or nn_xyz exists.", fixed = TRUE)
})
