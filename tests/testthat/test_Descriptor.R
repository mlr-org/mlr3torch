test_that("Descriptor basic checks", {
  descriptor = Descriptor$new(
    generator = nn_mse_loss,
    id = "mse",
    param_set = ps(reduction = p_uty()),
    packages = "R6",
    label = "MSE Loss",
    man = "torch::nn_mse_loss"
  )

  expect_identical(descriptor$generator, nn_mse_loss)
  expect_identical(descriptor$id, "mse")
  expect_identical(descriptor$param_set$ids(), "reduction")
  expect_set_equal(descriptor$packages, c("R6", "mlr3torch", "torch"))
  expect_identical(descriptor$man, "torch::nn_mse_loss")

  expect_class(descriptor, "Descriptor")

  observed = capture.output(descriptor)

  expected = c(
    "<Descriptor:mse> MSE Loss",
    "* Generator: nn_mse_loss",
    "* Parameters: list()",
    "* Packages: R6,torch,mlr3torch"
  )
  expect_identical(observed, expected)

  expect_error(Descriptor$new(
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
