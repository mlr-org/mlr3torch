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
  expect_equal(shape_to_str(1), "(1)")
  expect_equal(shape_to_str(c(1, 2)), "(1,2)")
  expect_equal(shape_to_str(NULL), "(<unknown>)")
  expect_error(shape_to_str("a"))

  # list
  expect_equal(shape_to_str(list(c(NA, 2), c(2, 3))), c("[(NA,2);(2,3)]"))

  md = po("torch_ingress_ltnsr")$train(list(nano_imagenet()))[[1L]]
})
