test_that("pot works", {
  # without id increment
  # for nn_
  obj = pot("linear")
  expect_r6(obj, "PipeOpTorchLinear")
  expect_true(obj$id == "linear")
  # for torch_optimizer
  obj = pot("optimizer", "adam")
  expect_r6(obj, "PipeOpTorchOptimizer")
  expect_true(obj$id == "optimizer")

  # with id increment
  # for nn_
  obj = pot("linear")
  expect_r6(obj, "PipeOpTorchLinear")
  expect_true(obj$id == "linear")
  # for torch_optimizer
  obj = pot("optimizer_90", "adam")
  expect_r6(obj, "PipeOpTorchOptimizer")
  expect_true(obj$id == "optimizer_90")
})

test_that("pot gives informative error message", {
  expect_error(pot("nn_linear"), regex = "You probably wanted po(\"nn_", fixed = TRUE)
  expect_error(pot("torch_optimizer"), regex = "You probably wanted po(\"torch_", fixed = TRUE)
})
