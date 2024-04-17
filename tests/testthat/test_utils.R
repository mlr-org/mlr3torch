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

test_that("get_nout works", {
  expect_equal(get_nout(tsk("iris")), 3)
  expect_equal(get_nout(tsk("mtcars")), 1)
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
