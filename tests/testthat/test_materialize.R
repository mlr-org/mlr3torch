test_that("materialize works", {
  lt = as_lazy_tensor(torch_randn(1, 3))
  lt1 = transform_lazy_tensor(lt, po("module", module = function(x) - abs(x)), shape = c(NA, 3))
  lt2 = transform_lazy_tensor(lt, po("module", module = function(x) abs(x)), shape = c(NA, 3))

  lt3 = c(lt1, lt2)

  expect_list(materialize(lt3, cat = FALSE))
  m = materialize(lt3, cat = TRUE)
  expect_class(m, "torch_tensor")
  expect_true(torch_equal(-m[1, ], m[2]))
  expect_class(m, "torch_tensor")
  expect_equal(dim(m), c(2, 3))

  d = data.table(x = lt, y = runif(10))
  expect_data_table(materialize(d, cat = FALSE))
  expect_list(materialize(d, cat = TRUE))
})
