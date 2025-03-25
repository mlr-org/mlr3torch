test_that("basic", {
  task = tsk("german_credit")
  ds = task_dataset(
    task = task,
    feature_ingress_tokens = list(x = TorchIngressToken(features = "age", batchgetter_num, c(NA, 1)))
  )

  expect_class(ds, "dataset")
  expect_equal(length(ds), task$nrow)
  batch = ds$.getbatch(1)
  expect_list(batch)
  expect_class(batch$x$x, "torch_tensor")
  expect_equal(batch$.index, torch_tensor(1, torch_long()))

  # now we check with two ingress tokens

  ingress_num = (po("select", selector = selector_type("integer")) %>>% po("torch_ingress_num"))$train(task)[[1L]]$ingress[[1L]] # nolint
  ingress_categ = (po("select", selector = selector_type("factor")) %>>% po("torch_ingress_categ"))$train(task)[[1L]]$ingress[[1L]]  # nolint

  # check that the device works
  ds = task_dataset(
    task = task,
    feature_ingress_tokens = list(x_num = ingress_num, x_categ = ingress_categ),
    target_batchgetter = get_target_batchgetter(task$task_type)
  )

  batch = ds$.getbatch(7)

  expect_permutation(names(batch), c("x", "y", ".index"))
  expect_equal(batch$.index, torch_tensor(7L))
  expect_equal(batch$y$shape, 1)
  expect_equal(batch$x$x_num$shape, c(1, ingress_num$shape[2]))
  expect_equal(batch$x$x_categ$shape, c(1, ingress_categ$shape[2]))

  batch2 = ds$.getbatch(7:8)
  expect_permutation(names(batch2), c("x", "y", ".index"))
  expect_equal(batch2$.index, torch_tensor(7:8))
  expect_equal(batch2$y$shape, 2)
  expect_equal(batch2$x$x_num$shape, c(2, ingress_num$shape[2]))
  expect_equal(batch2$x$x_categ$shape, c(2, ingress_categ$shape[2]))

  # input checks on ingress tokens
  expect_error(task_dataset(task, list()))
  expect_error(task_dataset(task, list(1)))
})

test_that("task_dataset throws error for empty ingress tokens", {
  expect_error(task_dataset(
    task = tsk("iris"),
    feature_ingress_tokens = list(x = TorchIngressToken(features = character(0), batchgetter_num, c(NA, 0)))
  ), regexp = "with no features")
})

test_that("task_dataset does not change when task is modified afterwards", {
  task = tsk("iris")
  ds = task_dataset(
    task = task,
    feature_ingress_tokens = list(x1 = TorchIngressToken(features = "Sepal.Length", batchgetter_num, c(NA, 1)))
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
    feature_ingress_tokens = list(x = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4)))
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
    target_batchgetter = get_target_batchgetter("regr")
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
    feature_ingress_tokens = list(x1 = TorchIngressToken(features = "Sepal.Length", batchgetter_num, c(NA, 1)))
  )

  batch = ds$.getbatch(1)

  expect_true(is.null(batch$y))
  expect_class(batch$x$x1, "torch_tensor")
})

test_that("batchgetter_num", {
  data = data.table(x_int = 1:3, x_dbl = runif(3))
  x = batchgetter_num(data)
  expect_class(x, "torch_tensor")
  expect_equal(x$shape, c(3, 2))
  expect_true(x$dtype == torch_float())
  expect_equal(x$device$type, "cpu")
})

test_that("dataset_num works for classification and regression", {
  # classification
  task = tsk("iris")$select(c("Sepal.Length", "Petal.Length"))$filter(1:2)
  ds = dataset_num(task, list(device = "cpu"))
  batch = ds$.getbatch(1:2)

  expected = torch_tensor(as.matrix(task$data(1:2, task$feature_names)))

  expect_true(torch_equal(torch_tensor(expected), batch$x[[1]]))
  expect_true(torch_equal(torch_tensor(expected), batch$x[[1]]))

  # regression
  ids = 2:4
  task = tsk("mtcars")$select(c("am", "carb", "cyl"))$filter(ids)
  ds = dataset_num(task, list(device = "cpu"))
  batch = ds$.getbatch(seq_len(length(ids)))

  expected = torch_tensor(as.matrix(task$data(2:4, task$feature_names)))
  expect_true(torch_equal(torch_tensor(expected), batch$x[[1]]))
})

test_that("batchgetter_categ works", {
  data = data.table(
    x_fct = factor(c("a", "b", "a", "b")),
    x_ord = ordered(c("1", "2", "3", "4")),
    x_lgl = c(TRUE, FALSE, TRUE, FALSE)
  )
  x = batchgetter_categ(data)
  expect_class(x, "torch_tensor")
  expect_true(x$dtype == torch_long())
  expect_equal(x$device$type, "cpu")
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
  y_loaded = target_batchgetter1(y)
  expect_equal(y_loaded$shape, c(5, 1))
  expect_class(y_loaded, "torch_tensor")
  expect_equal(y_loaded$device$type, "cpu")
})

test_that("default target batchgetter works: classification", {
  target_batchgetter2 = get_target_batchgetter("classif")
  y = data.table(y = factor(c("a", "b", "a", "b")))
  y_loaded = target_batchgetter2(y)
  expect_equal(y_loaded$shape, 4)
  expect_class(y_loaded, "torch_tensor")
  expect_equal(y_loaded$device$type, "cpu")
})

test_that("caching of graph", {
  env = new.env()
  fn = crate(function(x) {
    env$counter = env$counter + 1L
    x + 1
  }, env)
  po_test = pipeop_preproc_torch("test", fn = fn, shapes_out = "infer", stages_init = "both")$new()

  expect_true(is.function(po_test$fn))

  taskin = tsk("lazy_iris")

  graph = po_test %>>%
    list(
      po("trafo_nop_1") %>>% po("renamecolumns", renaming = c(x = "z")),
      po("trafo_nop_2")) %>>%
    po("featureunion")

  task = graph$train(taskin)[[1L]]

  ingress_tokens = list(
    z = TorchIngressToken("z", batchgetter_lazy_tensor, c(NA, 4)),
    x = TorchIngressToken("x", batchgetter_lazy_tensor, c(NA, 4))
  )

  ds = task_dataset(task, ingress_tokens)
  env$counter = 0L

  ds$.getbatch(1)
  expect_equal(env$counter, 1)
})

test_that("merging of graphs behaves as expected", {
  task = tsk("lazy_iris")

  e = new.env()
  e$a = 0
  fn = crate(function(x) {
    a <<- a + 1
    x
  }, .parent = e)
  graph = pipeop_preproc_torch("trafo_counter", fn = fn)$new() %>>%
    list(
      po("renamecolumns", renaming = c(x = "x1"), id = "a1") %>>% po("trafo_nop", id = "nop1"),
      po("renamecolumns", renaming = c(x = "x2"), id = "a2") %>>% po("trafo_nop", id = "nop2")
    )

  tasks = graph$train(task)
  task = tasks[[1]]

  task$cbind(tasks[[2]]$data(cols = "x2"))$filter(1:2)

  cols = task$data(cols = task$feature_names)

  ds = task_dataset(
    task,
    list(
      x1 = TorchIngressToken("x1", batchgetter_lazy_tensor, c(NA, 4)),
      x2 = TorchIngressToken("x2", batchgetter_lazy_tensor, c(NA, 4))
    )
  )

  ds$.getbatch(1)
  expect_true(e$a == 1L)

  # now we ensure that the addition of this unrelated columns (has a different dataset hash)
  # does not intefer with the merging of the other columns
  task$cbind(data.table(x3 = as_lazy_tensor(runif(150)), ..row_id = 1:150))

  ds2 = task_dataset(
    task,
    list(
      x1 = TorchIngressToken("x1", batchgetter_lazy_tensor, c(NA, 4)),
      x2 = TorchIngressToken("x2", batchgetter_lazy_tensor, c(NA, 4)),
      x3 = TorchIngressToken("x3", batchgetter_lazy_tensor, c(NA, 1))
    )
  )

  e$a = 0
  ds2$.getbatch(1)
  g1 = dd(ds2$task$data(cols = "x1")$x1)$graph
  g2 = dd(ds2$task$data(cols = "x2")$x2)$graph
  expect_true(identical(g1, g2))

  expect_true(e$a == 1L)
})

test_that("nop added for non-terminal pipeop", {
  po_add = PipeOpPreprocTorchAddSome$new()

  task = tsk("lazy_iris")
  graph = po("trafo_nop") %>>%
    list(
      po("renamecolumns", renaming = c(x = "y")) %>>% po("torch_ingress_ltnsr_1"),
      po_add %>>% po("torch_ingress_ltnsr_2")) %>>%
    po("nn_merge_sum")

  task = graph$train(task)[[1L]]$task

  res = materialize(task$data()[1, ], rbind = TRUE)
  expect_equal(res$y + 1, res$x)

})

test_that("unmergeable graphs are handled correctly", {
  ds = dataset(
    initialize = function() {
      self$x1 = 1:10
      self$x2 = 2:11
    },
    .getitem = function(i) {
      list(x1 = torch_tensor(self$x1[i]), x2 = torch_tensor(self$x2[1]))
    },
    .length = function() 10
  )()
  lt1 = as_lazy_tensor(ds, dataset_shapes = list(x1 = NULL, x2 = NULL), graph = po("nop", id = "nop1"),
    input_map = "x1", pointer_shape = c(NA, 1))
  lt2 = as_lazy_tensor(ds, dataset_shapes = list(x1 = NULL, x2 = NULL), graph = po("nop", id = "nop1"), input_map = "x2",
    pointer_shape = c(NA, 1))

  task = as_task_regr(data.table(y = runif(10), x1 = lt1, x2 = lt2), target = "y", id = "t1")

  i1 = po("torch_ingress_ltnsr_1")$train(list(task$clone(deep = TRUE)$select("x1")))[[1]]$ingress
  i2 = po("torch_ingress_ltnsr_2")$train(list(task$clone(deep = TRUE)$select("x2")))[[1]]$ingress
  expect_true(grepl(
    capture.output(invisible(task_dataset(task, c(i1, i2)))),
    pattern = "Cannot merge", fixed = TRUE
  ))
  prev_threshold = lg$threshold
  on.exit({lg$set_threshold(prev_threshold)}, add = TRUE) # nolint
  lg$set_threshold("error")
  task_ds = task_dataset(task, c(i1, i2))

  batch = task_ds$.getbatch(1)
  expect_equal(batch$x[[1]], batch$x[[2]] - 1)
})

test_that("merge_lazy_tensors only returns modified columns", {
  # don't waste memory by cbinding unmerged graph
  task = as_task_regr(data.table(
    y = 1:10,
    x1 = as_lazy_tensor(1:10),
    x2 = as_lazy_tensor(2:11)
  ), target = "y", id = "test")

  data = task$data(cols = task$feature_names)
  merged = merge_lazy_tensor_graphs(data)
  expect_true(is.null(merged))
})

test_that("y is NULL if no target batchgetter is provided", {
  task = tsk("lazy_iris")
  md = po("torch_ingress_ltnsr")$train(list(task))[[1L]]
  iter = dataloader_make_iter(dataloader(task_dataset(task, md$ingress,
    target_batchgetter = target_batchgetter_regr)))
  batch = iter$.next()
  expect_class(batch$y, "torch_tensor")

  iter = dataloader_make_iter(dataloader(task_dataset(task, md$ingress)))
  batch = iter$.next()
  expect_true(is.null(batch$y))
})

test_that("works with selector", {
  task = tsk("iris")
  ingress_token = ingress_num()

  ds = task_dataset(
    task = task,
    feature_ingress_tokens = list(features = ingress_token),
    target_batchgetter = get_target_batchgetter(task$task_type)
  )

  expect_equal(length(ds), task$nrow)

  batch_single = ds$.getbatch(1)

  # Check structure of the batch
  expect_list(batch_single)
  expect_list(batch_single$x)
  expect_class(batch_single$x$features, "torch_tensor")
  expect_class(batch_single$y, "torch_tensor")
  expect_equal(batch_single$.index, torch_tensor(1, dtype = torch_long()))

  # Check tensor properties
  expect_equal(batch_single$x$features$shape, c(1, 4))
  expect_true(batch_single$x$features$dtype == torch_float())

  # Check values - should match the first row of the iris dataset
  first_row = as.numeric(as.matrix(task$data(rows = 1, cols = task$feature_names)))
  tensor_values = as.numeric(batch_single$x$features)
  expect_equal(tensor_values, first_row, tolerance = 1e-6)

  # Test with a batch of samples
  batch_indices = 5:10
  batch_multiple = ds$.getbatch(batch_indices)

  # Check structure and shape
  expect_equal(batch_multiple$x$features$shape, c(length(batch_indices), 4))
  expect_equal(batch_multiple$.index, torch_tensor(batch_indices, dtype = torch_long()))

  # Check values - should match rows 5:10 of the iris dataset
  rows_data = as.matrix(task$data(rows = batch_indices, cols = task$feature_names))
  expect_equal(as.matrix(batch_multiple$x$features), unname(rows_data), tolerance = 1e-6)
})

test_that("task_dataset works with ingress_num, ingress_categ and ingress_ltnsr", {
  task = tsk("iris")

  ds = task_dataset(
    task = task,
    feature_ingress_tokens = list(x = ingress_num()),
    target_batchgetter = get_target_batchgetter(task$task_type)
  )

  # Test dataset length
  expect_equal(length(ds), task$nrow)

  # Get single sample batch
  batch_single = ds$.getbatch(2:1)

  expect_list(batch_single)
  expect_list(batch_single$x)
  expect_class(batch_single$y, "torch_tensor")
  expect_equal(batch_single$.index, torch_tensor(2:1, dtype = torch_long()))
  expect_class(batch_single$x$x, "torch_tensor")

  task = as_task_regr(
    data.table(y = 1:10, x1 = factor(letters[1:10]), x2 = c(TRUE, FALSE), x3 = ordered(1:10)),
    target = "y"
  )
  ds = task_dataset(task, feature_ingress_tokens = list(x = ingress_categ()))
  expect_equal(length(ds), task$nrow)
  batch = ds$.getbatch(2:1)
  expect_equal(batch$x$x$shape, c(2, 3))
  expect_true(batch$x$x$dtype == torch_long())
})
