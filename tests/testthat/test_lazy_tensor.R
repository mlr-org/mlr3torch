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

  expect_class(materialize(lazy_tensor()), "torch_tensor")
})

test_that("transform_lazy_tensor works", {
  lt = as_lazy_tensor(torch_randn(16, 2, 5))
  lt_mat = materialize(lt)

  expect_equal(lt_mat$shape, c(16, 2, 5))

  mod = nn_module(
    forward = function(x) {
      torch_reshape(x, c(-1, 10))
    }
  )()
  po_module = po("module", module = mod, id = "mod")

  new_shape = c(NA, 10)

  lt1 = transform_lazy_tensor(lt, po_module, new_shape)

  dd1 = attr(lt1, "data_descriptor")
  expect_equal(dd$graph$edges,
    data.table(src_id = "dataset_x", src_channel = "output", dst_id = "mod", dst_channel = "input")
  )

  dd = attr(lt, "data_descriptor")

  # graph was cloned
  expect_true(!identical(dd1$graph, dd$graph))
  # pipeop was cloned
  expect_true(!identical(dd1$graph$pipeops$dataset_x, dd$graph$pipeops$dataset_x))

  # .pointer was set
  expect_equal(dd1$.pointer, c("mod", "output"))

  # .pointer_shape was set
  expect_equal(dd1$.pointer_shape, c(NA, 10))

  # hash was updated
  expect_false(dd$.hash == dd1$.hash)
  expect_true(dd$.dataset_hash == dd1$.dataset_hash)

  # materialization gives correct result
  lt1_mat = materialize(lt1)
  expect_equal(lt1_mat$shape, c(16, 10))

  lt1_mat = torch_reshape(lt1_mat, c(-1, 2, 5))
  expect_true(torch_equal(lt1_mat, lt_mat))
})
