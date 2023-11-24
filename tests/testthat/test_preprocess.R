test_that("trafo_resize", {
  autotest_pipeop_torch_preprocess(
    obj = po("trafo_resize",  size = c(3, 4)),
    shapes_in = list(c(16, 10, 10, 4), c(3, 4, 8))
  )
})
