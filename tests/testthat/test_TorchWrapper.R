test_that("TorchWrapper basic checks", {
  wrapper = TorchWrapper$new(
    generator = nn_mse_loss,
    id = "mse",
    param_set = ps(reduction = p_uty()),
    packages = "R6",
    label = "MSE Loss",
    man = "torch::nn_mse_loss"
  )

  expect_identical(wrapper$generator, nn_mse_loss)
  expect_identical(wrapper$id, "mse")
  expect_identical(wrapper$param_set$ids(), "reduction")
  expect_set_equal(wrapper$packages, c("R6", "mlr3torch", "torch"))
  expect_identical(wrapper$man, "torch::nn_mse_loss")

  expect_class(wrapper, "TorchWrapper")

  observed = capture.output(wrapper)

  expected = c(
    "<TorchWrapper:mse> MSE Loss",
    "* Generator: nn_mse_loss",
    "* Parameters: list()",
    "* Packages: R6,torch,mlr3torch"
  )
  expect_identical(observed, expected)

  expect_error(TorchWrapper$new(
    generator = nn_mse_loss,
    id = "mse",
    param_set = ps(reduction = p_uty(), x = p_uty()),
    packages = "R6",
    label = "MSE Loss",
    man = "torch::nn_mse_loss"
    ),
    regexp = "Parameter values with ids 'x' are missing in generator.",
    fixed = TRUE
    )

})
