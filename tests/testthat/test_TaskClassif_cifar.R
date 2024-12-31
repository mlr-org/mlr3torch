skip_on_cran()

test_that("CIFAR-10 works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("cifar10")

  expect_equal(task$nrow, 50000)

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

  # check the responses
  train_idx = 1:50000
  int_mlr3torch_responses = as.integer(task$data()$class[train_idx])

  cifar10_ds_train = cifar10_dataset(root = file.path(get_cache_dir(), "datasets", "cifar10", "raw"), train = TRUE,
    download = FALSE)

  # TODO: determine whether a separate function is truly necessary
  # proably not, if getitem() allows a vector
  get_response = function(idx, ds) {
    ds$.getitem(idx)$y
  }
  cifar10_ds_responses = map_int(train_idx, get_response, ds = cifar10_ds_train)

  expect_true(all.equal(int_mlr3torch_responses, cifar10_ds_responses))
  
  # check a subset of train images
  small_train_idx = c(1, 2, 27, 9999,
    10000, 10001, 10901, 19999,
    20000, 20001, 29999,
    30000, 30001, 39999,
    40000, 40001, 49999,
    50000
  )
  task_small = task$clone()
  task_small$filter(small_train_idx)

  test_same_at_idx = function(idx, lt_list, imgs_arr) {
    all.equal(as.array(lt_list[[idx]]), imgs_arr[idx, , , ])
  }

  lt_list = materialize(task_small$data()$image)
  imgs_arr = cifar10_ds_train$.getitem(small_train_idx)$x

  expect_true(all(map_lgl(1:length(small_train_idx), test_same_at_idx, lt_list = lt_list, imgs_arr = imgs_arr)))

  # check a subset of test images
  test_idx = c(1, 2, 27, 8484, 9999, 10000)
  
  test_dt_from_task = task$backend$data(rows = 50001:60000, cols = task$backend$colnames)
  expect_true(all(test_dt_from_task$split == "test"))

  lt_list_test = materialize(test_dt_from_task[test_idx, ]$image)

  cifar10_ds_test = cifar10_dataset(root = file.path(get_cache_dir(), "datasets", "cifar10", "raw"), train = FALSE,
    download = FALSE)
  imgs_arr_test = cifar10_ds_test$.getitem(test_idx)$x

  expect_true(all(map_lgl(1:length(test_idx), test_same_at_idx, lt_list = lt_list, imgs_arr = imgs_arr)))
})