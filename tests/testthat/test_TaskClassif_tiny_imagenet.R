skip_on_cran()

test_that("tiny_imagenet works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("tiny_imagenet")

  expect_equal(task$nrow, 110000)
  # this makes the test faster
  task$filter(1:10)
  expect_equal(task$id, "tiny_imagenet")
  expect_equal(task$label, "ImageNet Subset")
  expect_equal(task$feature_names, "image")
  expect_equal(task$target_names, "class")
  expect_equal(task$man, "mlr3torch::mlr_tasks_tiny_imagenet")
  task$data()
  expect_true("tiny-imagenet-200" %in% list.files(file.path(get_cache_dir(), "datasets", "tiny_imagenet", "raw")))
  expect_true("data.rds" %in% list.files(file.path(get_cache_dir(), "datasets", "tiny_imagenet")))
  expect_equal(task$backend$nrow, 120000)
  expect_equal(task$backend$ncol, 4)
})
