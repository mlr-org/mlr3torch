test_that("imageuri, works", {
  uris = list.files(system.file("toytask", "images", package = "mlr3torch"), full.name = TRUE)
  vec = imageuri(uris)
  expect_true(inherits(vec, "imageuri"))
})
