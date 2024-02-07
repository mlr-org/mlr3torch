test_that("DataBackendLazy works", {
  n = 5
  constructor = function(backend) {
    dt = data.table(
      y = c(0.1, 0.2, 0.3, 0.4, 0.5),
      x_fct = factor(letters[1:n]),
      x_num = c(2, 4, 6, 8, 10),
      row_id = n:1
    )

    DataBackendDataTable$new(data = dt, primary_key = "row_id")
  }

  column_info = data.table(
    id = c("y", "x_fct", "x_num", "row_id"),
    type = c("numeric", "factor", "numeric", "integer"),
    levels = list(NULL, letters[1:n], NULL, NULL)
  )

  backend_lazy = DataBackendLazy$new(
    constructor = constructor,
    rownames = n:1,
    col_info = column_info,
    primary_key = "row_id",
    data_formats = "data.table"
  )

  expect_r6(backend_lazy, c("DataBackend", "DataBackendLazy"))
  expect_equal(backend_lazy$nrow, n)
  expect_equal(backend_lazy$ncol, 4)
  # Order is not guaranteed
  expect_equal(backend_lazy$colnames, c("y", "x_fct", "x_num", "row_id"))
  expect_equal(capture.output(backend_lazy),
    c("<DataBackendLazy> (5x4)",
      " * Backend not loaded yet.")
  )
  expect_permutation(backend_lazy$rownames, n:1)
  expect_false(backend_lazy$is_constructed)

  expect_r6(backend_lazy$backend, c("DataBackend", "DataBackendDataTable"))
  expect_true(backend_lazy$is_constructed)

  expect_true(any(grepl(capture.output(backend_lazy), pattern = "x_fct")))

  expect_equal(backend_lazy$data(rows = 1, cols = "y")[[1L]], 0.5)
  expect_equal(backend_lazy$data(rows = 2, cols = "y")[[1L]], 0.4)
  expect_equal(backend_lazy$data(rows = 3, cols = "x_num")[[1L]], 6)

  expect_the_same = function(f) {
    expect_equal(f(backend_lazy), f(backend_lazy$backend))
  }

  expect_the_same(function(x) x$head(2))
  expect_the_same(function(x) x$distinct(1:2, c("y", "x_fct")))
  expect_the_same(function(x) x$missings(1:5, c("y", "x_fct", "x_num")))
  expect_the_same(function(x) x$data(c(2, 3), c("y", "row_id")))
  expect_true(test_equal_col_info(col_info(backend_lazy), col_info(backend_lazy$backend)))

  constructor_constructor = function(colnames = letters[1:3], rownames = 1:10, letter = "a") {
    ncol = length(colnames)

    function(backend) {
      cols = lapply(colnames[1:(length(colnames) - 1)], function(x) {
        dt = data.table(rownames)
        names(dt) = x
        dt
      })
      dt = data.table(factor(rep(letter, times = length(rownames))))
      names(dt) = colnames[length(colnames)]
      cols[colnames[ncol]] = dt
      dt = do.call(cbind, args = cols)
      as_data_backend(dt, primary_key = "a")
    }
  }

  # Here, we always pass the same meta information during construction
  # The arguments only modify the constructor (via constructor_constructor)
  # We do this to test that the checks are correctly performed after constructing the backend
  # (the checks that verify that the meta information was specified correctly)

  expect_correct_error = function(
      colnames = letters[1:3],
      rownames = 1:10,
      regexp,
      letter = "a",
      fixed = TRUE
    ) {
    constructor = constructor_constructor(colnames, rownames, letter)

    col_info = data.table(
      id = c("a", "b", "c"),
      type = c("integer", "integer", "factor"),
      levels = list(NULL, NULL, "a")
    )

    expect_error(DataBackendLazy$new(
        constructor = constructor,
        rownames = 1:10,
        col_info = col_info,
        primary_key = "a",
        data_formats = "data.table"
      )$backend,
      regexp = regexp
    )
  }
  expect_correct_error(regexp = NA)
  expect_correct_error(colnames = c("a", toupper(letters[2:3])), regexp = "column names")
  expect_correct_error(rownames = 1:100, regexp = "row identifiers")
  expect_correct_error(letter = "b", regexp = "column info")
})

test_that("primary_key must be in col_info", {
  expect_error(DataBackendLazy$new(
    constructor = function(backend) NULL,
    col_info = data.table(id = "a", type = "integer", levels = list(NULL)),
    rownames = 1,
    primary_key = "b",
    data_formats = "data.table"
  ), regexp = "Must be element of")
})

test_that("primary_key must be the same for backends", {
  constructor = function(backend) {
    DataBackendDataTable$new(
      data.table(y = 1:5, x = 1:5),
      primary_key = "x"
    )
  }
  backend_lazy = DataBackendLazy$new(
    constructor = constructor,
    col_info = data.table(id = c("x", "y"), type = rep("integer", 2), levels = list(NULL, NULL)),
    rownames = 1:5,
    primary_key = "y",
    data_formats = "data.table"
  )
  expect_error(backend_lazy$backend, "primary key")
})

test_that("constructor must have argument backend", {
  expect_error(DataBackendLazy$new(
    constructor = function() NULL,
    col_info = data.table(id = c("x", "y"), type = rep("integer", 2), levels = list(NULL, NULL)),
    rownames = 1:5,
    primary_key = "y",
    data_formats = "data.table"
  ), regexp = "formal arguments")
})
