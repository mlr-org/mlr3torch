test_that("nano_mnist", {
  task = nano_mnist()
  x = materialize(task$data(cols = task$feature_names)[[1L]], rbind = TRUE)
  expect_equal(x$shape, c(10, 1, 28, 28))

  xout = materialize(po("augment_vflip")$train(list(task))[[1L]]$data(cols = "image")[[1L]], rbind = TRUE)
  expect_equal(xout$shape, c(10, 1, 28, 28))
})

test_that("nano_imagenet", {
  task = nano_imagenet()
  task$row_roles$use = c(2, 3)
  x = materialize(task$data(cols = task$feature_names)[[1L]], rbind = TRUE)
  expect_equal(x$shape, c(2, 3, 64, 64))

  xout = materialize(po("augment_vflip")$train(list(task))[[1L]]$data(cols = "image")[[1L]], rbind = TRUE)
  expect_equal(xout$shape, c(2, 3, 64, 64))
})

test_that("nano_dogs_vs_cats", {
  task = nano_dogs_vs_cats()
  task$row_roles$use = c(2, 3)
  x = materialize(task$data(cols = task$feature_names)[[1L]], rbind = FALSE)
  expect_list(x, types = "torch_tensor")
})
