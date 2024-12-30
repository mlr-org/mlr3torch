skip_on_cran()

test_that("CIFAR-10 works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("cifar10")

  expect_equal(task$nrow, 50000)

  # idx_to_test = c(1, 2, 27, 9999,
#   10000, 10001, 10901, 19999,
#   20000, 20001, 29999,
#   30000, 30001, 39999,
#   40000, 40001, 49999,
#   50000)

  task$filter(1:10)
  expect_equal(task$id, "cifar10")
  expect_equal(task$label, "CIFAR-10 Classification")
  expect_equal(task$feature_names, "image")
  expect_equal(task$target_names, "class")
  expect_equal(task$man, "mlr3torch::mlr_tasks_cifar10")
  task$data()
  expect_true("cifar-10-batches-bin" %in% list.files(file.path(get_cache_dir(), "datasets", "cifar10", "raw")))
  expect_true("data.rds" %in% list.files(file.path(get_cache_dir(), "datasets", "cifar10")))
  expect_equal(task$backend$nrow, 60000)
  expect_equal(task$backend$ncol, 6)
})

test_that("CIFAR-10 data matches the torchvision implementation", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("cifar10")
  task$data()

  cifar10_ds_train = cifar10_dataset(root = file.path(get_cache_dir(), "datasets", "cifar10", "raw"), train = TRUE,
    download = FALSE)

  train_idx = 1:50000
  int_mlr3torch_responses = as.integer(task$class[trn_idx])

  get_response = function(idx, ds) {
    ds$.getitem(idx)$y
  }
  int_tv_responses = map_int(trn_idx, get_response, ds = tv_cifar10_ds)

  all.equal(int_mlr3torch_responses, int_tv_responses)

  test_same_at_idx = function(idx, lt_col, ds) {
    all.equal(as.array(lt_col[[idx]]), ds$.getitem(idx)$x)
  }
  
  small_train_idx = c(1, 2, 27, 9999,
    10000, 10001, 10901, 19999,
    20000, 20001, 29999,
    30000, 30001, 39999,
    40000, 40001, 49999,
    50000
  )
  task$filter(train_idx)
  task$data()

  all(map_lgl(.x = small_train_idx, .f = test_same_at_idx, lt = task$image, ds = cifar10_ds_train))

  train_idx = c(1, 2, 27, 9999,
    10000, 10001, 10901, 19999,
    20000, 20001, 29999,
    30000, 30001, 39999,
    40000, 40001, 49999,
    50000)

  all(map_lgl(.x = idx_to_test, .f = test_same_at_idx, ds_mlr3torch = cifar10_ds, ds_torch = tv_cifar10_ds))

  test_same_at_idx(10001, cifar10_ds, tv_cifar10_ds)


  test_idx = c(50001, 50002, 58484, 59999, 60000)

})