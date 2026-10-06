test_that("auto_device handles NULL and 'auto'", {
  # NULL should be returned as-is
  expect_null(auto_device(NULL))
  # explicit device should be returned as-is
  expect_equal(auto_device("cpu"), "cpu")
  # "auto" should return either "cuda" or "cpu"
  expect_true(auto_device("auto") %in% c("cuda", "cpu"))
})

test_that("make_check_vector works", {
  check_vector1 = make_check_vector(1)
  expect_true(check_vector1(1))
  expect_equal(check_vector1(1:2), "Must be an integerish vector of length 1.")

  check_vector2 = make_check_vector(2)
  expect_true(check_vector2(1:2))
  expect_equal(check_vector2(1:3), "Must be an integerish vector of length 1 or 2.")
})

test_that("test_equal_col_info works", {
  ci = data.table(id = "x", type = "factor", levels = list(c("a", "b")))
  expect_true(test_equal_col_info(ci, ci))

  ci1 = data.table(id = "y", type = "factor", levels = list(c("a", "b")))
  expect_false(test_equal_col_info(ci, ci1))
  ci2 = data.table(id = "x", type = "ordered", levels = list(c("a", "b")))
  expect_false(test_equal_col_info(ci, ci2))
  ci3 = data.table(id = "x", type = "factor", levels = list(c("b", "a")))
  expect_false(test_equal_col_info(ci, ci3))
  ci4 = data.table(id = "x", type = "factor", levels = list("a"))
  expect_false(test_equal_col_info(ci, ci4))
})

test_that("uniqueify works", {
  expect_equal(uniqueify("a", "a"), "a_1")
})

test_that("auto_cache_lazy_tensors", {
  ds = random_dataset(3)
  ds2 = random_dataset(3)

  # 1) Duplicated dataset_hash
  x1 = list(
    as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, 3)), ids = 1:3),
    as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, 3)), ids = 1:3)
  )
  expect_true(auto_cache_lazy_tensors(x1))

  # 2) Duplicated hash
  x2 = list(
    as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, 3)), ids = 1:3),
    as_lazy_tensor(ds2, dataset_shapes = list(x = c(NA, 3)), ids = 1:3)
  )
  expect_false(auto_cache_lazy_tensors(x2))
})

test_that("order_named_args works", {
  expect_equal(list(x = 1, y = 2), order_named_args(function(x, y) NULL, list(y = 2, x = 1)))
  expect_equal(list(x = 1, y = 2), order_named_args(function(x, y) NULL, list(y = 2, 1)))
  expect_equal(list(x = 1, y = 2), order_named_args(function(x, y) NULL, list(x = 1, 2)))
  expect_equal(list(x = 1, 2, 3), order_named_args(function(x, ...) NULL, list(2, 3, x = 1)))
  expect_equal(list(1, 2, 3), order_named_args(function(...) NULL, list(1, 2, 3)))
  expect_equal(order_named_args(function(..., x) NULL, list(1, 2)), list(1, 2))
  # no way to pass specfied argument correctly by position, everything would be eaten by ...
  expect_error(order_named_args(function(..., x) NULL, list(2, 3, x = 1)), regexp = "`...` must")
  expect_error(order_named_args(function(y, ..., x) NULL, list(y = 4, 2, 3, x = 1)), regexp = "`...` must")
})

test_that("shape_to_str works", {
  expect_equal(shape_to_str(c(NA, NA)), "(NA,NA)")
  expect_equal(shape_to_str(1), "(1)")
  expect_equal(shape_to_str(c(1, 2)), "(1,2)")
  expect_equal(shape_to_str(NULL), "(<unknown>)")
  expect_error(shape_to_str("a"))

  # list
  expect_equal(shape_to_str(list(c(NA, 2), c(2, 3))), c("[(NA,2);(2,3)]"))

  md = po("torch_ingress_ltnsr")$train(list(nano_imagenet()))[[1L]]
})


test_that("auto_device() rejects cuda when it is unavailable", {
  skip_if(cuda_is_available(), "CUDA is available")
  expect_error(auto_device("cuda"), "no CUDA device is available")
  expect_equal(auto_device("auto"), "cpu")
  expect_equal(auto_device("cpu"), "cpu")
  expect_null(auto_device(NULL))
})

test_that("rbind_arrays binds along the first dimension", {
  a = array(1:12, c(2L, 3L, 2L))
  b = array(13:30, c(3L, 3L, 2L))
  out = rbind_arrays(list(a, b))

  expect_equal(dim(out), c(5L, 3L, 2L))
  # every observation keeps the slice it came with, in the order the elements were given
  expect_equal(out[1:2, , ], a)
  expect_equal(out[3:5, , ], b)

  # `rbind()` itself only understands two dimensions and would flatten the rest into columns,
  # which is the whole reason this exists
  expect_equal(dim(rbind(a, a)), c(2L, 12L))
})

test_that("rbind_arrays handles the degenerate shapes", {
  # one element is returned as it is
  a = array(1:12, c(2L, 3L, 2L))
  expect_equal(rbind_arrays(list(a)), a)

  # a one-dimensional array has nothing to rotate around
  expect_equal(rbind_arrays(list(array(1:2), array(3:4))), array(1:4))

  # an element without observations contributes nothing
  empty = array(integer(0), c(0L, 3L, 2L))
  expect_equal(rbind_arrays(list(empty, a)), a)
  expect_equal(dim(rbind_arrays(list(empty, empty))), c(0L, 3L, 2L))

  # matrices work, they are just arrays with two dimensions -- but unlike `rbind()` the dimnames
  # are dropped, which is why `pt_combine()` still uses `rbind()` for them
  m = matrix(1:4, nrow = 2L, dimnames = list(NULL, c("a", "b")))
  expect_equal(rbind_arrays(list(m, m)), matrix(c(1:2, 1:2, 3:4, 3:4), nrow = 4L))
  expect_null(colnames(rbind_arrays(list(m, m))))
})

test_that("rbind_arrays rejects arrays that differ beyond the first dimension", {
  expect_error(
    rbind_arrays(list(array(1:12, c(2L, 3L, 2L)), array(1:8, c(2L, 2L, 2L)))),
    "differ beyond the first dimension", fixed = TRUE
  )
  # a different number of dimensions is caught by the same check
  expect_error(
    rbind_arrays(list(array(1:12, c(2L, 3L, 2L)), matrix(1:6, nrow = 2L))),
    "differ beyond the first dimension", fixed = TRUE
  )
})

test_that("rbind_arrays keeps the storage type of its elements", {
  expect_type(rbind_arrays(list(array(1L, c(1L, 2L)), array(2L, c(1L, 2L)))), "integer")
  expect_type(rbind_arrays(list(array(1.5, c(1L, 2L)), array(2.5, c(1L, 2L)))), "double")
  expect_type(rbind_arrays(list(array(TRUE, c(1L, 2L)), array(FALSE, c(1L, 2L)))), "logical")
  expect_type(rbind_arrays(list(array("a", c(1L, 2L)), array("b", c(1L, 2L)))), "character")
})
