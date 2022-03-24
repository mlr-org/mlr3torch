test_that("Architecture is working", {
  architecture = top("linear") %>>%
    top("relu")
})
