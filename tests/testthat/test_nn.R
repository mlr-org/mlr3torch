test_that("nn works", {
  x = nn("linear", out_features = 3)
  expect_equal(x$id, "linear")
  expect_class(x, "PipeOpTorchLinear")
  expect_equal(x$param_set$values$out_features, 3)
})
