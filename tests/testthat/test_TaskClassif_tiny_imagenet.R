skip_on_cran()

test_that("tiny_imagenet task works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("tiny_imagenet")

  dt = task$data()
  expect_equal(task$backend$nrow, 120000)
  expect_equal(task$backend$ncol, 4)
  expect_data_table(dt, ncols = 2, nrows = 100000)
  expect_permutation(colnames(dt), c("class", "image"))
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