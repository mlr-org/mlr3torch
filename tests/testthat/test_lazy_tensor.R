test_that("prototype", {
  proto = lazy_tensor()
  expect_class(proto, "lazy_tensor")
  expect_true(length(proto) == 0L)
  expect_error(dd(proto))

  expect_error(materialize(lazy_tensor()), "Cannot materialize")
})

test_that("input checks", {
  desc = dd(as_lazy_tensor(1:10))
  expect_error(lazy_tensor(desc, 1:11), "ids")
  expect_error(lazy_tensor(desc, NA), "missing values")
})


test_that("unknown shapes", {
  ds = dataset(
    initialize = function() {
      self$x = list(
        torch_randn(2, 2),
        torch_randn(3, 3)
      )
    },
    .getitem = function(i) {
      list(x = self$x[[i]])
    },
    .length = function() {
      length(self$x)
    }
  )()

  dd = DataDescriptor$new(ds, dataset_shapes = list(x = NULL))
  expect_class(dd, "DataDescriptor")
  expect_equal(dd$pointer_shape, NULL)
  expect_equal(dd$dataset_shapes, list(x = NULL))

  lt = as_lazy_tensor(dd)
  expect_class(lt, "lazy_tensor")
  expect_error(materialize(lt), regexp = NA)
  expect_true(test_class(lt[1:2], "lazy_tensor"))
  expect_class(lt[[1]], "list")

  expect_class(c(lt, lazy_tensor()), "lazy_tensor")
})

test_that("assignment", {
  x = as_lazy_tensor(1:2)
  x[2] = x[1]
  expect_class(x, "lazy_tensor")
  expect_equal(
    materialize(x[1], rbind = TRUE),
    materialize(x[2], rbind = TRUE)
  )
  # cannot assign beyond vector length
  expect_error({x[3] = as_lazy_tensor(1)}, "max") # nolint
  expect_error({x[2] = 1}, "class") # nolint
  # indices must be ints
  expect_error({x["hallo"] = as_lazy_tensor(1)}, "integerish") # nolint
  expect_error({x[1] = as_lazy_tensor(10)}, "data descriptor") # nolint


  y = lazy_tensor()
  y[[1]] = as_lazy_tensor(1)
  expect_class(y, "lazy_tensor")
  expect_true(length(y) == 1)

  expect_error({x[1] = as_lazy_tensor(1)}) # nolint
})


test_that("concatenation", {
  x1 = as_lazy_tensor(1)
  x2 = as_lazy_tensor(1)
  expect_error(c(x1, x2), regexp = "Can only")

  x = c(x1, x1)
  expect_class(x, "lazy_tensor")
  expect_equal(length(x), 2)

  # can still concatenate lazy tensor with other objects
  l = list(1, x)
  expect_class(l, "list")
  expect_false(inherits(l, "lazy_tensor"))
})

test_that("subsetting and indexing", {
  x = as_lazy_tensor(1:3)
  expect_class(x[1:2], "lazy_tensor")
  expect_false(inherits(x[[1]], "lazy_tensor"))
  expect_equal(length(x[1:2]), 2)
  expect_list(x[[1]], len = 2L)
  expect_class(x[integer(0)], "lazy_tensor")
  expect_equal(length(x[integer(0)]), 0)
})


test_that("transform_lazy_tensor", {
  lt = as_lazy_tensor(torch_randn(16, 2, 5))
  lt_mat = materialize(lt, rbind = TRUE)

  expect_equal(lt_mat$shape, c(16, 2, 5))

  mod = nn_module(
    forward = function(x) {
      torch_reshape(x, c(-1, 10))
    }
  )()
  po_module = po("module", module = mod, id = "mod")

  new_shape = c(NA, 10)

  lt1 = transform_lazy_tensor(lt, po_module, new_shape)

  dd1 = dd(lt1)

  expect_equal(dd1$graph$edges,
    data.table(src_id = dd1$graph$input$op.id, src_channel = "output", dst_id = "mod", dst_channel = "input")
  )

  dd = dd(lt)

  # graph was cloned
  expect_true(!identical(dd1$graph, dd$graph))
  # pipeop was not cloned
  expect_true(identical(dd1$graph$pipeops$dataset_x, dd$graph$pipeops$dataset_x))

  # pointer was set
  expect_equal(dd1$pointer, c("mod", "output"))

  # pointer_shape was set
  expect_equal(dd1$pointer_shape, c(NA, 10))

  # hash was updated
  expect_false(dd$hash == dd1$hash)
  expect_true(dd$dataset_hash == dd1$dataset_hash)

  # materialization gives correct result
  lt1_mat = materialize(lt1, rbind = TRUE)
  expect_equal(lt1_mat$shape, c(16, 10))

  lt1_mat = torch_reshape(lt1_mat, c(-1, 2, 5))
  expect_true(torch_equal(lt1_mat, lt_mat))
})

test_that("pofu identifies identical columns", {
  dt = data.table(
    y = 1:2,
    z = as_lazy_tensor(1:2)
  )

  taskin = as_task_regr(dt, target = "y", id = "test")

  po_fu = po("featureunion")
  taskout = po_fu$train(list(taskin, taskin))[[1L]]

  expect_set_equal(taskout$feature_names, "z")
})

test_that("as_lazy_tensor for dataset", {
  ds = random_dataset(3)
  x = as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, 3)), ids = 1:5)
  expect_class(x, "lazy_tensor")
  expect_equal(ds$.getbatch(1:5)$x, materialize(x, rbind = TRUE))
})

test_that("as_lazy_tensor for DataDescriptor", {
  ds = random_dataset(3)
  dd = DataDescriptor$new(
    dataset = ds,
    dataset_shapes = list(x = c(NA, 3))
  )
  x = as_lazy_tensor(dd, ids = c(1, 5))
  expect_class(x, "lazy_tensor")
  expect_equal(ds$.getbatch(c(1, 5))$x, materialize(x, rbind = TRUE))
})

test_that("as_lazy_tensor for tensor", {
  tnsr = torch_randn(10, 1)
  x = as_lazy_tensor(tnsr)
  expect_class(x, "lazy_tensor")
  expect_equal(tnsr, materialize(x, rbind = TRUE))

})

test_that("as_lazy_tensor for numeric", {
  x = as_lazy_tensor(1:10)
  expect_class(x, "lazy_tensor")
  expect_equal(1:10, as.numeric(as_array(materialize(x, rbind = TRUE))))
})

test_that("format", {
  expect_equal(format(lazy_tensor()), character(0))
  expect_equal(format(as_lazy_tensor(1)), "<tnsr[1]>")
  expect_equal(format(as_lazy_tensor(1:2)), c("<tnsr[1]>", "<tnsr[1]>"))
})

test_that("printer", {
  expect_equal(
    capture.output(as_lazy_tensor(1:2)),
    c("<ltnsr[2]>", "[1] <tnsr[1]> <tnsr[1]>")
  )
})

test_that("comparison", {
  x = as_lazy_tensor(1:2)
  # diffe
  y = as_lazy_tensor(1:2)
  expect_equal(x == x, c(TRUE, TRUE))
  expect_equal(x[2:1] == x, c(FALSE, FALSE))
  expect_equal(x[c(1, 1)] == x, c(TRUE, FALSE))
  expect_equal(x == y, c(FALSE, FALSE))
})

test_that("error messages: no torch tensor or no unique names", {

  ds = dataset(
    initialize = function() self$x = torch_randn(10, 3, 3),
    .getitem = function(i) list(x = self$x[i, ], y = sample.int(1)),
    .length = function() nrow(self$x)
  )()

  expect_error(
    as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, 3, 3), y = NULL)),
    regexp = "must return torch tensors"
  )

  dsb = dataset(
    initialize = function() self$x = torch_randn(10, 3, 3),
    .getbatch = function(i) list(x = self$x[i, , drop = FALSE], y = sample.int(1)),
    .length = function() nrow(self$x)
  )()

  expect_error(
    as_lazy_tensor(dsb, dataset_shapes = list(x = c(NA, 3, 3), y = NULL)),
    regexp = "must return torch tensors"
  )

  ds1 = dataset(
    initialize = function() self$x = torch_randn(10, 3, 3),
    .getitem = function(i) list(self$x[i, ]),
    .length = function() nrow(self$x)
  )()
  expect_error(
    as_lazy_tensor(ds1, dataset_shapes = list(x = c(NA, 3, 3))),
    regexp = "list with named elements"
  )

  ds1b = dataset(
    initialize = function() self$x = torch_randn(10, 3, 3),
    .getbatch = function(i) list(self$x[i, drop = FALSE]),
    .length = function() nrow(self$x)
  )()
  expect_error(
    as_lazy_tensor(ds1b, dataset_shapes = list(x = c(NA, 3, 3))),
    regexp = "list with named elements"
  )
})

test_that("recycling in data.table", {
  d = data.table(x = 1:2, y = as_lazy_tensor(1))
  expect_class(d$y, "lazy_tensor")
})

test_that("rep for lazy_tensor", {
  expect_equal(
    materialize(rep(as_lazy_tensor(c(1, 2)), times = 2), rbind = TRUE),
    torch_tensor(matrix(c(1, 2, 1, 2), ncol = 1))
  )
  expect_equal(
    materialize(rep(as_lazy_tensor(c(1, 2)), each = 2), rbind = TRUE),
    torch_tensor(matrix(c(1, 1, 2, 2), ncol = 1))
  )
})

test_that("rep_len for lazy_tensor", {
  expect_equal(
    materialize(rep_len(as_lazy_tensor(c(1, 2)), length.out = 3), rbind = TRUE),
    torch_tensor(matrix(c(1, 2, 1), ncol = 1))
  )
})
