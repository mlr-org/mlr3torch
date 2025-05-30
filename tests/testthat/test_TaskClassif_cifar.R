skip_on_cran()

test_that("CIFAR-10 works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("cifar10")

  expect_equal(task$nrow, 60000)

  task$filter(1:10)
  expect_equal(task$id, "cifar10")
  expect_equal(task$label, "CIFAR-10 Classification")
  expect_equal(task$feature_names, "image")
  expect_equal(task$target_names, "class")
  expect_equal(task$man, "mlr3torch::mlr_tasks_cifar")
  expect_equal(task$backend$hash, "mlr3torch::mlr_tasks_cifar10")
  task$data()
  expect_true("cifar-10-batches-bin" %in% list.files(file.path(get_cache_dir(), "datasets", "cifar10", "raw")))
  expect_true("data.rds" %in% list.files(file.path(get_cache_dir(), "datasets", "cifar10")))
  expect_equal(task$backend$nrow, 60000)
  expect_equal(task$backend$ncol, 4)
})

test_that("CIFAR-100 works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("cifar100")

  expect_equal(task$nrow, 60000)

  task$filter(1:10)
  expect_equal(task$id, "cifar100")
  expect_equal(task$label, "CIFAR-100 Classification")
  expect_equal(task$feature_names, "image")
  expect_equal(task$target_names, "class")
  expect_equal(task$man, "mlr3torch::mlr_tasks_cifar")
  expect_equal(task$backend$hash, "mlr3torch::mlr_tasks_cifar100")
  task$data()
  expect_true("cifar-100-binary" %in% list.files(file.path(get_cache_dir(), "datasets", "cifar100", "raw")))
  expect_true("data.rds" %in% list.files(file.path(get_cache_dir(), "datasets", "cifar100")))
  expect_equal(task$backend$nrow, 60000)
  expect_equal(task$backend$ncol, 4)
})