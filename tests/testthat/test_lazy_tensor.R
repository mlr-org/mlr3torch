test_that("lazy_tensor works", {
  dd1 = DataDescriptor(random_dataset(5, 4), list(x = c(NA, 5, 4)))
  dd2 = DataDescriptor(random_dataset(5, 4), list(x = c(NA, 5, 4)))

  lt = lazy_tensor()
  expect_class(lt, "lazy_tensor")
  expect_true(length(lt) == 0L)
  expect_true(is.null(attr(lt, "data_descriptor")))

  lt1 = lazy_tensor(dd1)
  lt2 = lazy_tensor(dd2)

  expect_error(c(lt1, lt2), "attributes are incompatible")

  lt1_empty = lt1[integer(0)]
  expect_true(is.null(attr(lt1_empty, "data_descriptor")))
  expect_class(lt1_empty, "lazy_tensor")

  lt1_empty[1] = lt1[1]
})
