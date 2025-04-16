test_that("correct input checks", {

})

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
  expect_data_table(tbl, nrow = 3, ncol = 3)
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
  # test following example pipeops:
  # - target trafo
  # - fix factors
  # - smote

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

    expect_equal(materialize(tbl$x, rbind = TRUE), ds$x[rows])
    expect_equal(tbl$y, as.integer(ds$y[rows]))
  }
  check(be, ds, 1, c("x", "y"), 1)
  # y is no in the cache, so .getitem() is not called on $data()
  check(be, ds, 1, "y", 0)

  # but x is not cached, so we still need to call .getitem below
  check(be, ds, 1, c("x", "y"), 1)

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
  ds = tensor_dataset(
    x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))$float(),
    y = torch_tensor(as.matrix(1:100, nrow = 100, ncol = 1))$float()
  )

  be = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)),
    converter = list(y = as.numeric))
  task = as_task_regr(be, target = "y")

  learner = lrn("regr.mlp", epochs = 200, batch_size = 100, jit_trace = TRUE, opt.lr = 1, seed = 1)
  rr = resample(task, learner, rsmp("insample"))
  expect_true(rr$aggregate(msr("regr.rmse")) < 3)
})

test_that("can train a binary classification learner", {
  ds = tensor_dataset(
    x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))$float(),
    y = torch_tensor(as.matrix(1:100, nrow = 100, ncol = 1))$float()
  )

  be = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)),
    converter = list(y = as.numeric))
  task = as_task_regr(be, target = "y")

  learner = lrn("regr.mlp", epochs = 200, batch_size = 100, jit_trace = TRUE, opt.lr = 1, seed = 1)
  rr = resample(task, learner, rsmp("insample"))
  expect_true(rr$aggregate(msr("regr.rmse")) < 3)
})

test_that("can train a multiclass classification learner", {
  ds = tensor_dataset(
    x = torch_tensor(matrix(100:1, nrow = 100, ncol = 1))$float(),
    y = torch_tensor(matrix(rep(c(0, 1), each = 50), nrow = 100, ncol = 1))$float()
  )

  be = as_data_backend(ds, dataset_shapes = list(x = c(NA, 1), y = c(NA, 1)),
    converter = list(y = function(x) factor(as.integer(x), levels = c(0, 1), labels = c("yes", "no"))))
  task = as_task_classif(be, target = "y")

  learner = lrn("classif.mlp", epochs = 200, batch_size = 100, jit_trace = TRUE, opt.lr = 1, seed = 1)
  rr = resample(task, learner, rsmp("insample"))
  expect_true(rr$aggregate(msr("regr.rmse")) < 3)
})