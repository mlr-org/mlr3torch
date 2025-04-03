test_that("multi_tensor_dataset works with .getitem", {
  ds1 = dataset(
    initialize = function() {
      self$x1 = torch_randn(100, 10)
      self$x2 = torch_randn(100, 5)
      self$y = torch_randn(100, 1)
    },
    .getitem = function(i) {
      list(
        x = list(x1 = self$x1[i], x2 = self$x2[i]),
        y = self$y[i],
        .index = i
      )
    },
    .length = function() {
      nrow(self$x1)
    }
  )()
  ds2 = multi_tensor_dataset(ds1)
  f = function(b1, b2) {
    expect_equal(b1$x$x1$unsqueeze(1), b2$x$x1)
    expect_equal(b1$x$x2$unsqueeze(1), b2$x$x2)
    expect_equal(b1$y$unsqueeze(1), b2$y)
    expect_equal(b1$.index, b2$.index)
  }
  f(ds1$.getitem(1), ds2$.getbatch(1))
  f(ds1$.getitem(100), ds2$.getbatch(100))
  expect_equal(length(ds2), length(ds1))
  ds3 = multi_tensor_dataset(ds1, device = "meta")
  b1 = ds3$.getbatch(1)
  expect_true(b1$x$x1$device == torch_device("meta"))
  expect_true(b1$x$x2$device == torch_device("meta"))
  expect_true(b1$y$device == torch_device("meta"))
})


test_that("multi_tensor_dataset works with .getbatch", {
  ds1 = dataset(
    initialize = function() {
      self$x1 = torch_randn(100, 10)
      self$x2 = torch_randn(100, 5)
      self$y = torch_randn(100, 1)
    },
    .getbatch = function(i) {
      list(
        x = list(x1 = self$x1[i, drop = FALSE], x2 = self$x2[i, drop = FALSE]),
        y = self$y[i, drop = FALSE],
        .index = i
      )
    },
    .length = function() {
      nrow(self$x1)
    }
  )()
  ds2 = multi_tensor_dataset(ds1)
  expect_equal(ds1$.getbatch(1:2), ds2$.getbatch(1:2))
  expect_equal(ds1$.getbatch(3:2), ds2$.getbatch(3:2))
  expect_equal(length(ds2), length(ds1))
  ds3 = multi_tensor_dataset(ds1, device = "meta")
  b1 = ds3$.getbatch(1)
  expect_true(b1$x$x1$device == torch_device("meta"))
  expect_true(b1$x$x2$device == torch_device("meta"))
  expect_true(b1$y$device == torch_device("meta"))
})
