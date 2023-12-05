test_that("Unknown shapes work", {
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

  dd = DataDescriptor(ds, dataset_shapes = list(x = NULL))
  expect_class(dd, "DataDescriptor")
  expect_equal(dd$.pointer_shape, NULL)
  expect_equal(dd$dataset_shapes, list(x = NULL))

  lt = as_lazy_tensor(dd)
  expect_class(lt, "lazy_tensor")
  materialize(lt)

  ds = random_dataset(10, 3)
  expect_error(DataDescriptor(ds, list(x = NULL)))
})

test_that("lazy_tensor works", {
  dd1 = DataDescriptor(random_dataset(5, 4), list(x = c(NA, 5, 4)))
  dd2 = DataDescriptor(random_dataset(5, 4), list(x = c(NA, 5, 4)))

  lt = lazy_tensor()
  expect_class(lt, "lazy_tensor")
  expect_true(length(lt) == 0L)
  expect_error(is.null(lt$data_descriptor))

  lt1 = lazy_tensor(dd1)
  lt2 = lazy_tensor(dd2)

  expect_error(c(lt1, lt2), "attributes are incompatible")

  expect_error({lt1[1] = lt2[1]}) # nolint

  lt1_empty = lt1[integer(0)]
  expect_error(is.null(lt1_empty$data_descriptor))
  expect_class(lt1_empty, "lazy_tensor")

  expect_error(materialize(lazy_tensor()), "Cannot materialize")
})

test_that("transform_lazy_tensor works", {
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

  dd1 = lt1$data_descriptor

  expect_equal(dd1$graph$edges,
    data.table(src_id = "dataset_x", src_channel = "output", dst_id = "mod", dst_channel = "input")
  )

  dd = lt$data_descriptor

  # graph was cloned
  expect_true(!identical(dd1$graph, dd$graph))
  # pipeop was not cloned
  expect_true(identical(dd1$graph$pipeops$dataset_x, dd$graph$pipeops$dataset_x))

  # .pointer was set
  expect_equal(dd1$.pointer, c("mod", "output"))

  # .pointer_shape was set
  expect_equal(dd1$.pointer_shape, c(NA, 10))

  # hash was updated
  expect_false(dd$.hash == dd1$.hash)
  expect_true(dd$.dataset_hash == dd1$.dataset_hash)

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
