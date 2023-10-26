test_that("task_dataset basic tests", {
  task = tsk("german_credit")
  ds = task_dataset(
    task = task,
    feature_ingress_tokens = list(x = TorchIngressToken(features = "age", batchgetter_num, c(NA, 1))),
    device = "cpu"
  )

  expect_class(ds, "dataset")
  expect_equal(ds$device, "cpu")
  expect_equal(length(ds), task$nrow)
  batch = ds$.getbatch(1)
  expect_list(batch)
  expect_class(batch$x$x, "torch_tensor")
  expect_equal(batch$.index, 1)

  # now we check with two ingress tokens

  ingress_num = (po("select", selector = selector_type("integer")) %>>% po("torch_ingress_num"))$train(task)[[1L]]$ingress[[1L]] # nolint
  ingress_categ = (po("select", selector = selector_type("factor")) %>>% po("torch_ingress_categ"))$train(task)[[1L]]$ingress[[1L]]  # nolint

  # check that the device works
  ds = task_dataset(
    task = task,
    feature_ingress_tokens = list(x_num = ingress_num, x_categ = ingress_categ),
    target_batchgetter = get_target_batchgetter(task$task_type),
    device = "cpu"
  )

  batch = ds$.getbatch(7)

  expect_permutation(names(batch), c("x", "y", ".index"))
  expect_equal(batch$.index, 7)
  expect_equal(batch$y$shape, 1)
  expect_equal(batch$x$x_num$shape, c(1, ingress_num$shape[2]))
  expect_equal(batch$x$x_categ$shape, c(1, ingress_categ$shape[2]))

  batch2 = ds$.getbatch(7:8)
  expect_permutation(names(batch2), c("x", "y", ".index"))
  expect_equal(batch2$.index, 7:8)
  expect_equal(batch2$y$shape, 2)
  expect_equal(batch2$x$x_num$shape, c(2, ingress_num$shape[2]))
  expect_equal(batch2$x$x_categ$shape, c(2, ingress_categ$shape[2]))
})

test_that("task_dataset throws error for empty ingress tokens", {
  expect_error(task_dataset(
    task = tsk("iris"),
    feature_ingress_tokens = list(x = TorchIngressToken(features = character(0), batchgetter_num, c(NA, 0))),
    device = "cpu"
  ), regexp = "with no features")
})

test_that("task_dataset does not change when task is modified afterwards", {
  task = tsk("iris")
  ds = task_dataset(
    task = task,
    feature_ingress_tokens = list(x1 = TorchIngressToken(features = "Sepal.Length", batchgetter_num, c(NA, 1))),
    device = "cpu"
  )
  expect_equal(length(ds), 150)
  task$row_roles$use = 1
  expect_equal(length(ds), 150)
})

test_that("task_dataset only uses the rows with role 'use'", {
  task = tsk("iris")
  use = sample(task$row_ids, 3)
  task$row_roles$use = use

  ds = task_dataset(
    task = task,
    feature_ingress_tokens = list(x = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4))),
    device = "cpu"
  )

  expect_equal(length(ds), length(use))
})

test_that("task_dataset returns the correct data (even with non-standard row ids)", {
  dat = data.table(x = 1:3, row_id = c(20L, 10L, 30L), y = 3:1)
  backend = as_data_backend(dat, primary_key = "row_id")

  task = as_task_regr(backend, target = "y")

  ds = task_dataset(
    task = task,
    feature_ingress_tokens = list(x1 = TorchIngressToken(features = "x", batchgetter_num, c(NA, 1))),
    target_batchgetter = get_target_batchgetter("regr"),
    device = "cpu"
  )

  # the first batch should be row_id 2 of the task, whose x value is 1 and y value is 3

  f = function(i) {
    batch = ds$.getbatch(i)
    expect_equal(as.numeric(batch$y), task$data(task$row_ids[i], cols = "y")[[1L]][1L])
    expect_equal(as.numeric(batch$x$x1), task$data(task$row_ids[i], cols = "x")[[1L]][1L])
  }

  f(1)
  f(2)
  f(3)
})

test_that("task_datset works without a target", {
  ds = task_dataset(
    task = tsk("iris"),
    feature_ingress_tokens = list(x1 = TorchIngressToken(features = "Sepal.Length", batchgetter_num, c(NA, 1))),
    device = "cpu"
  )

  batch = ds$.getbatch(1)

  expect_true(is.null(batch$y))
  expect_class(batch$x$x1, "torch_tensor")
})

test_that("task_dataset respects the device", {
  f = function(device) {
    ds = task_dataset(
      task = tsk("iris"),
      feature_ingress_tokens = list(x1 = TorchIngressToken(features = "Sepal.Length", batchgetter_num, c(NA, 1))),
      target_batchgetter = get_target_batchgetter("classif"),
      device = device
    )

    batch = ds$.getbatch(1)

    expect_equal(batch$x$x1$device$type, device)
    expect_equal(batch$y$device$type, device)
  }

  f("cpu")
  f("meta")
})

test_that("batchgetter_num works", {
  data = data.table(x_int = 1:3, x_dbl = runif(3))
  x = batchgetter_num(data, "cpu")
  expect_class(x, "torch_tensor")
  expect_equal(x$shape, c(3, 2))
  expect_true(x$dtype == torch_float())
  expect_equal(x$device$type, "cpu")

  x1 = batchgetter_num(data, "meta")
  expect_class(x1, "torch_tensor")
  expect_equal(x1$shape, c(3, 2))
  expect_equal(x1$device$type, "meta")
})

test_that("dataset_num works for classification and regression", {
  # classification
  task = tsk("iris")$select(c("Sepal.Length", "Petal.Length"))$filter(1:2)
  ds = dataset_num(task, list(device = "cpu"))
  batch = ds$.getbatch(1:2)

  expected = torch_tensor(as.matrix(task$data(1:2, task$feature_names)))

  expect_true(torch_equal(torch_tensor(expected), batch$x$input))
  expect_true(torch_equal(torch_tensor(expected), batch$x$input))

  # regression
  ids = 2:4
  task = tsk("mtcars")$select(c("am", "carb", "cyl"))$filter(ids)
  ds = dataset_num(task, list(device = "cpu"))
  batch = ds$.getbatch(seq_len(length(ids)))

  expected = torch_tensor(as.matrix(task$data(2:4, task$feature_names)))
  expect_true(torch_equal(torch_tensor(expected), batch$x$input))
})

test_that("batchgetter_categ works", {
  data = data.table(
    x_fct = factor(c("a", "b", "a", "b")),
    x_ord = ordered(c("1", "2", "3", "4")),
    x_lgl = c(TRUE, FALSE, TRUE, FALSE)
  )
  x = batchgetter_categ(data, "cpu")
  expect_class(x, "torch_tensor")
  expect_true(x$dtype == torch_long())
  expect_equal(x$device$type, "cpu")

  x1 = batchgetter_categ(data, "meta")
  expect_equal(x1$device$type, "meta")
})

test_that("dataset_num_categ works for classification and regression", {
  task = tsk("german_credit")
  ds = dataset_num_categ(task, list(device = "cpu"))

  batch = ds$.getbatch(10:14)

  expect_permutation(names(batch$x), c("input_num", "input_categ"))
  expect_equal(batch$x$input_num$shape, c(5, sum(task$feature_types$type %in% c("numeric", "integer"))))
  expect_equal(batch$x$input_categ$shape, c(5, sum(task$feature_types$type %in% c("logical", "factor", "ordered"))))

  # test on the target
  expect_equal(batch$y$shape, 5)
  expect_true(batch$y$dtype == torch_long())

  # test what happens when there are no categoricals / numerics

  task_num = task$clone()
  task_num$select(c("age", "amount", "duration"))
  ds_num = dataset_num_categ(task_num, list(device = "cpu"))
  batch_num = ds_num$.getbatch(10:14)
  expect_true(is.null(batch_num$x$input_categ))

  # to the same for categoricals
  task_categ = task$clone()
  task_categ$select(c("credit_history", "employment_duration", "number_credits"))
  ds_categ = dataset_num_categ(task_categ, list(device = "cpu"))
  batch_categ = ds_categ$.getbatch(10:14)
  expect_true(is.null(batch_categ$x$input_num))
})

#test_that("target_batchgetter works for classification", {
#  task = tsk("iris")
#  ds = task_dataset(
#    task = task,
#    feature_ingress_tokens = list(x = TorchIngressToken(features = task$feature_names, batchgetter_num, c(NA, 4))),
#    target_batchgetter = get_target_batchgetter("classif"),
#    device = "cpu"
#  )
#
#  ids = task$row_ids[task$truth() != "Setosa"]
#  task$filter(ids)
#
#  expect_permutation(task$col_info["Species", "levels", on = "id"][[1L]][1L][1L], c("setosa", "versicolor", "virginica"))
#
#  fct1 = factor(1:3, levels = 3:1, labels = letters[3:1])
#  fct2 = factor(1:3, levels = 1:3, labels = letters[1:3])
#
#  expect_true(all(fct1 == fct2))
#
#  as.integer(fct1) == as.integer(fct2)
#})

test_that("default target batchgetter works: regression", {
  target_batchgetter1 = get_target_batchgetter("regr")
  y = data.table(y = 1:5)
  y_loaded = target_batchgetter1(y, "cpu")
  expect_equal(y_loaded$shape, c(5, 1))
  expect_class(y_loaded, "torch_tensor")
  expect_equal(y_loaded$device$type, "cpu")

  y_loaded1 = target_batchgetter1(y, "meta")
  expect_equal(y_loaded1$device$type, "meta")
})

test_that("default target batchgetter works: classification", {
  target_batchgetter2 = get_target_batchgetter("classif")
  y = data.table(y = factor(c("a", "b", "a", "b")))
  y_loaded = target_batchgetter2(y, "cpu")
  expect_equal(y_loaded$shape, 4)
  expect_class(y_loaded, "torch_tensor")
  expect_equal(y_loaded$device$type, "cpu")

  y_loaded1 = target_batchgetter2(y, "meta")
  expect_equal(y_loaded1$device$type, "meta")
})

test_that("get_batchgetter_img works", {
  task = nano_imagenet()
  batchgetter = get_batchgetter_img(imgshape = c(3, 64, 64))

  dat = task$data(1:2, task$feature_names)

  batch = batchgetter(dat, "cpu")
  expect_equal(batch$shape, c(2, 3, 64, 64))
  expect_equal(batch$device$type, "cpu")
  expect_true(batch$dtype == torch_float())

  batchgetter_wrong = get_batchgetter_img(imgshape = c(64, 64))
  expect_error(batchgetter_wrong(dat), regexp = "imgshape")

  batch1 = batchgetter(dat, "meta")
  expect_true(batch1$device$type == "meta")
})

test_that("dataset_img works", {
  task = nano_imagenet()
  ds = dataset_img(task, list(device = "cpu", channels = 3, height = 64, width = 64))

  batch = ds$.getbatch(1:2)
  expect_equal(batch$x$image$shape, c(2, 3, 64, 64))
  expect_equal(batch$x$image$device$type, "cpu")
  expect_true(batch$x$image$dtype == torch_float())

  ds_meta = dataset_img(task, list(device = "meta", channels = 3, height = 64, width = 64))
  batch_meta = ds_meta$.getbatch(1)
  expect_true(batch_meta$x$image$device$type == "meta")
})

