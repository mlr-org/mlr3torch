test_that("All activation functions can be constructred", {
  for (act in torch_reflections$activation) {
    expect_r6(top("activation", .activation = act), "TorchOp")
  }
})
