test_that("nn works", {
  x = nn("linear", out_features = 3)
  expect_equal(x$id, "linear")
  expect_class(x, "PipeOpTorchLinear")
  expect_equal(x$param_set$values$out_features, 3)
})

test_that("works with unnamed arguments", {
  obj = nn("block", nn("linear"))
  expect_equal(obj$id, "block")
  expect_equal(obj$block$pipeops[[1]]$id, "linear")
})
