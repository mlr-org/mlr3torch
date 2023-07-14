skip_on_cran()

test_that("tiny_imagenet task works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("tiny_imagenet")

  expect_error(task$data(), regexp = NA)
  expect_class(task, "TaskClassif")
  expect_equal(length(task$row_roles$use), 100000)
  expect_equal(length(task$row_roles$test), 10000)
  expect_equal(length(task$row_roles$holdout), 10000)
  expect_equal(task$feature_names, "image")
  expect_equal(task$target_names, "class")
  expect_equal(task$properties, "multiclass")
  expect_equal(task$id, "tiny_imagenet")
  expect_equal(task$label, "ImageNet Subset")
})
