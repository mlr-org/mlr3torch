test_that("main API works", {
  # regression target
  ds = tensor_dataset(
    x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1)),
    y = torch_tensor(as.matrix(1:100, nrow = 100, ncol = 1))
  )

  be = as_data_backend(ds, converter = list(y = as.numeric), dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)))

  # converted data

  batch1 = be$data(1, c("x", "y"))
  expect_class(batch1$x, "lazy_tensor")
  expect_equal(length(batch1$x), 1)
  expect_equal(materialize(batch1$x, rbind = TRUE), torch_tensor(matrix(100L, nrow = 1, ncol = 1)))
  expect_equal(batch1$y, 1)

  batch2 = be$data(2:1, c("x", "y"))
  expect_class(batch2$x, "lazy_tensor")
  expect_equal(length(batch2$x), 2)
  expect_equal(materialize(batch2$x, rbind = TRUE), torch_tensor(matrix(100:99, nrow = 2, ncol = 1)))
  expect_equal(batch2$y, c(2, 1))

  # lt data
  batch_lt1 = withr::with_options(list(mlr3torch.data_loading = TRUE), {
    be$data(1, c("x", "y"))
  })
  expect_class(batch_lt1$x, "lazy_tensor")
  expect_equal(length(batch_lt1$x), 1)
  expect_equal(materialize(batch_lt1$x, rbind = TRUE), torch_tensor(matrix(100L, nrow = 1, ncol = 1)))
  # y is still a lazy tensor
  expect_class(batch_lt1$y, "lazy_tensor")
  expect_equal(length(batch_lt1$y), 1)

  batch_lt2 = withr::with_options(list(mlr3torch.data_loading = TRUE), {
    be$data(2:1, c("x", "y"))
  })
  expect_class(batch_lt2$x, "lazy_tensor")
  expect_equal(length(batch_lt2$x), 2)
  expect_equal(materialize(batch_lt2$x, rbind = TRUE), torch_tensor(matrix(100:99, nrow = 2, ncol = 1)))
  # y is still a lazy tensor
  expect_class(batch_lt2$y, "lazy_tensor")
  expect_equal(length(batch_lt2$y), 2)

  # missings
  expect_equal(be$missings(1:100, c("y", "x")), c(y = 0, x = 0))
  expect_equal(be$missings(1:100, "y"), c(y = 0))
  expect_equal(be$missings(1:100, "x"), c(x = 0))

  # head
  tbl = be$head(n = 3)
  expect_data_table(tbl, nrows = 3, ncols = 3)
  expect_class(tbl$x, "lazy_tensor")
  expect_equal(materialize(tbl$x, rbind = TRUE), torch_tensor(matrix(100:98, nrow = 3, ncol = 1)))
  expect_class(tbl$y, "numeric")
  expect_equal(tbl$row_id, as.numeric(1:3))
  expect_class(tbl$row_id, "integer")
  expect_equal(tbl$row_id, 1:3)

  # distinct values: this can be expensive
  dist = be$distinct(1:3, c("x", "y", "row_id"))
  expect_list(dist, len = 3)
  expect_equal(materialize(dist$x, rbind = TRUE), torch_tensor(matrix(100:98, nrow = 3, ncol = 1)))
  expect_equal(dist$y, c(1, 2, 3))
  expect_equal(dist$row_id, 1:3)
})

test_that("classif target works", {
  ds = dataset(
    initialize = function() {
      self$x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))
      self$y = torch_tensor(matrix(rep(c(0, 1), each = 50), nrow = 100, ncol = 1))
    },
    .getitem = function(i) {
      list(x = self$x[i], y = self$y[i])
    },
    .length = function() {
      nrow(self$x)
    }
  )()

  tbl = as_lazy_tensors(ds, list(x = c(NA, 1), y = c(NA, 1)))
  tbl$row_id = 1:100

  be = DataBackendLazyTensors$new(tbl, primary_key = "row_id", converter = list(
    y = function(x) factor(as.integer(x), levels = c(0, 1), labels = c("yes", "no"))
  ))
  batch = be$data(c(1, 2, 51, 52), c("x", "y", "row_id"))
  expect_class(batch$y, "factor")
  expect_equal(batch$y, factor(c("yes", "yes", "no", "no"), levels = c("yes", "no")))

  batch_lt = withr::with_options(list(mlr3torch.data_loading = TRUE), {
    be$data(c(1, 2, 51, 52), c("x", "y", "row_id"))
  })
  expect_class(batch_lt$y, "lazy_tensor")
  expect_equal(length(batch_lt$y), 4)
  expect_equal(materialize(batch_lt$y, rbind = TRUE), torch_tensor(matrix(c(1, 1, 0, 0), nrow = 4, ncol = 1)))
})

test_that("errors when weird preprocessing", {
})

test_that("chunking works ", {
  ds = dataset(
    initialize = function() {
      self$x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))
      self$y = torch_tensor(as.matrix(1:100, nrow = 100, ncol = 1))
      self$counter = 0
    },
    .getbatch = function(i) {
      self$counter = self$counter + 1
      list(x = self$x[i, drop = FALSE], y = self$y[i, drop = FALSE])
    },
    .length = function() {
      nrow(self$x)
    }
  )()

  be = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)), chunk_size = 3,
    converter = list(y = as.numeric))

  counter_prev = ds$counter
  be$data(1:3, c("x", "y"))
  expect_equal(ds$counter, counter_prev + 1)
  counter_prev = ds$counter
  be$data(4:10, c("x", "y"))
  expect_equal(ds$counter, counter_prev + 3)
})

test_that("can retrieve 0 rows", {
  ds = tensor_dataset(
    x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1)),
    y = torch_tensor(as.matrix(1:100, nrow = 100, ncol = 1))
  )
  be = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)),
    converter = list(y = as.numeric))
  res = be$data(integer(0), c("x", "y", "row_id"))
  expect_data_table(res, nrows = 0, ncols = 3)
  expect_class(res$x, "lazy_tensor")
  expect_class(res$y, "numeric")
  expect_equal(res$row_id, integer(0))
})

test_that("task converters work", {
  # regression target
  ds = tensor_dataset(
    x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))$float(),
    y = torch_tensor(as.matrix(1:100, nrow = 100, ncol = 1))$float()
  )
  task = as_task_regr(ds, target = "y", converter = list(y = as.numeric))
  task$data(integer(0))
  expect_equal(task$head(2)$y, 1:2)
  expect_equal(task$feature_names, "x")
  expect_equal(task$target_names, "y")
  expect_task(task)


  # binary classification
  ds = tensor_dataset(
    x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))$float(),
    y = torch_tensor(rep(0:1, times = 50))$float()$unsqueeze(2L)
  )
  task = as_task_classif(ds, target = "y", levels = c("yes", "no"))
  expect_task(task)
  expect_equal(task$head()$y, factor(rep(c("yes", "no"), times = 3), levels = c("yes", "no")))
})

test_that("caching works", {
  dsc = dataset(
    initialize = function() {
      self$x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))
      self$y = torch_tensor(as.matrix(1:100, nrow = 100, ncol = 1))
      self$counter = 0
    },
    .getitem = function(i) {
      self$counter = self$counter + 1
      list(x = self$x[i], y = self$y[i])
    },
    .length = function() {
      nrow(self$x)
    }
  )

  ds = dsc()

  be = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)),
    converter = list(y = as.integer), cache = "y")

  check = function(be, ds, rows, cols, n) {
    counter_prev = ds$counter
    tbl = be$data(rows, cols)
    observed_n = ds$counter - counter_prev
    expect_equal(observed_n, n)

    if ("x" %in% cols) {
      expect_equal(materialize(tbl$x, rbind = TRUE), ds$x[rows])
    }
    if ("y" %in% cols) {
      expect_equal(tbl$y, as.integer(ds$y[rows]))
    }
  }
  check(be, ds, 1, c("x", "y"), 1)
  # y is no in the cache, so .getitem() is not called on $data()
  check(be, ds, 1, "y", 0)

  # everything is in the cache
  check(be, ds, 1, c("x", "y"), 0)
  # lazy tensor causes no materialization
  check(be, ds, 1, "x", 0)

  # more than one row also works
  check(be, ds, 2:1, "y", 1)
  check(be, ds, c(3, 1), "y", 1)
  check(be, ds, 1:3, "y", 0)

  # when caching more than one, we materialize only once per batch
  be2 = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)),
    converter = list(y = as.integer, x = as.integer), cache = c("y", "x"))

  check2 = function(be, ds, rows, cols, n) {
    counter_prev = ds$counter
    tbl = be$data(rows, cols)
    observed_n = ds$counter - counter_prev
    expect_equal(observed_n, n)

    expect_equal(tbl$y, as.integer(ds$y[rows]))
    expect_equal(tbl$x, as.integer(ds$x[rows]))
  }

  check2(be2, ds, 1, c("x", "y"), 1)
  check2(be2, ds, 1, c("x", "y"), 0)
  check2(be2, ds, 2:1, c("x", "y"), 1)
  check2(be2, ds, 2, c("x", "y"), 0)
})

test_that("can train a regression learner", {
  x = torch_randn(100, 1)
  y = x + torch_randn(100, 1)
  ds = tensor_dataset(
    x = x,
    y = y
  )

  be = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)),
    converter = list(y = as.numeric))
  task = as_task_regr(be, target = "y")

  learner = lrn("regr.mlp", epochs = 10, batch_size = 100, jit_trace = TRUE, opt.lr = 1, seed = 1)
  rr = resample(task, learner, rsmp("insample"))
  expect_true(rr$aggregate(msr("regr.rmse")) < 1.5)
})

test_that("can train a binary classification learner", {
  ds = tensor_dataset(
    x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))$float(),
    y = torch_tensor(rep(0:1, each = 50))$float()$unsqueeze(2L)
  )

  be = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)),
    converter = list(y = function(x) factor(as.integer(x), levels = c(1, 0), labels = c("yes", "no"))))
  task = as_task_classif(be, target = "y")

  learner = lrn("classif.mlp", epochs = 10, batch_size = 100, jit_trace = TRUE, opt.lr = 10, seed = 1)
  rr = resample(task, learner, rsmp("insample"))
  expect_true(rr$aggregate(msr("classif.ce")) < 0.1)
})

test_that("can train a multiclass classification learner", {
  ds = tensor_dataset(
    x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))$float(),
    y = torch_tensor(rep(1:4, each = 25))
  )

  be = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = NA),
    converter = list(y = function(x) factor(as.integer(x), levels = 1:4, labels = c("a", "b", "c", "d"))))
  task = as_task_classif(be, target = "y")

  learner = lrn("classif.mlp", epochs = 10, batch_size = 100, jit_trace = TRUE, opt.lr = 0.2, seed = 1,
    neurons = 100)
  rr = resample(task, learner, rsmp("insample"))
  # just ensures that we lear something
  expect_true(rr$aggregate(msr("classif.ce")) < 0.6)
})

test_that("check_lazy_tensors_backend works", {
  ds = tensor_dataset(
    x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))$float(),
    y = torch_tensor(as.matrix(1:100, nrow = 100, ncol = 1))$float()
  )

  be = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)),
    converter = list(y = as.numeric))
  task_orig = as_task_regr(be, target = "y")

  expect_error(check_lazy_tensors_backend(task_orig$backend, c("x", "y")),
    regexp = NA)

  task1 = task_orig$clone(deep = TRUE)$cbind(data.table(y = 1:100))
  expect_error(check_lazy_tensors_backend(task1$backend, c("x", "y")),
    regexp = "A converter column ('y')", fixed = TRUE)

  task2 = task_orig$clone(deep = TRUE)$rbind(data.table(x = as_lazy_tensor(1), y = 2, row_id = 999))
  expect_error(check_lazy_tensors_backend(task2$backend, c("x", "y")),
    regexp = "A converter column ('y')", fixed = TRUE)
})


test_that("...", {
  ds = dataset(
    initialize = function(x, y) {
      self$x = torch_randn(100, 3)
      self$y = torch_randn(100, 1)
      self$counter = 0
    },
    .getbatch = function(i) {
      print("hallo")
      self$counter = self$counter + 1L
      list(x = self$x[i, drop = FALSE], y = self$y[i, drop = FALSE])
    },
    .length = function() 100
  )()

task = as_task_regr(ds, target = "y")

counter = ds$counter
task$head()
print(ds$counter - counter)
counter = ds$counter
task$head()
expec
print(ds$counter - counter)

})
