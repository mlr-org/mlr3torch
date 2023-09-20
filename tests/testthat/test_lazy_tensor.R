test_that("lazy_tensor works", {
  ds = dataset(
    initialize = function() {
      self$x = torch_randn(10, 5, 3)
    },
    .getitem = function(i) {
      list(x = self$x[i, ..])
    },
    .length = function() {
      nrow(self$x)
    }
  )()

  dd = DataDescriptor(ds, dataset_shapes = list(x = c(NA, 5, 3)))

  ltnsr0 = lazy_tensor()
  expect_class(ltnsr0, "lazy_tensor")

  ltnsr = as_lazy_tensor(dd)

  expect_equal(length(ltnsr), 10)
  expect_class(ltnsr, "lazy_tensor")

  ltnsr1 = c(ltnsr, ltnsr)
  expect_equal(length(ltnsr1), 20)
  expect_class(ltnsr1, "lazy_tensor")

  expect_class(vctrs::vec_cast(unclass(ltnsr), lazy_tensor()), "lazy_tensor")

  x = torch_randn(10, 3)
  ltnsr2 = as_lazy_tensor(x)
  expect_class(ltnsr2, "lazy_tensor")

  expect_equal(as_lazy_tensor(NULL), lazy_tensor())
  expect_class(as_lazy_tensor(ds, list(x = c(NA, 5, 3))), "lazy_tensor")
})

test_that("subsetting to empty lazy_tensor drops dimension", {
  lt = as_lazy_tensor(ds, list(x = c(NA, 5, 3)))
  expect_true(is.null(lt[integer()]$shape))
})

test_that("lazy_tensors must have same shape", {
  ds = random_dataset(5, 4, n = 10)

  lt1 = as_lazy_tensor(torch_randn(10, 3))
  lt2 = as_lazy_tensor(torch_randn(10, 4))

  expect_error(c(lt1, lt2), "must be equal")

  expect_error({lt1[1L] <- lt2[1]}, "Cannot cast")
})

test_that("transform_lazy_tensor works", {
  ds = random_dataset(5, 4, n = 10)

  lt = as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, 5, 4)))

  expect_equal(data.table::address(lt[[1L]]$data_descriptor), data.table::address(lt[[2L]]$data_descriptor))

  po_resize = po("module", id = "resize", module = function(x) torch_resize(x, c(-1, 20)))
  lt1 = transform_lazy_tensor(lt, po_resize, c(NA, 20))

  expect_true(length(lt[[1L]]$data_descriptor$graph$pipeops) == 1L)
  expect_true(length(lt1[[1L]]$data_descriptor$graph$pipeops) == 2L)

  expect_true("resize" %in% names(lt1[[1L]]$data_descriptor$graph$pipeops))

  expect_equal(
    data.table(src_id = "random_dataset_x", src_channel = "output", dst_id = "resize", dst_channel = "input"),
    lt1[[1L]]$data_descriptor$graph$edges
  )

  expect_false(data.table::address(lt1[[1L]]$data_descriptor$graph$pipeops$lazy_transform) == data.table::address(po_resize))
  expect_equal(data.table::address(lt1[[1L]]$data_descriptor), data.table::address(lt1[[2L]]$data_descriptor))
  expect_true(lt1[[1L]]$id == lt[[1L]]$id)
  expect_true(lt1[[2L]]$id == lt[[2L]]$id)

  # now a more complicated lazy tensor with two different DataDescripors
  po_add = po("module", id = "add_10", module = function(x) x + 10)

  ds2 = random_dataset(5, 4, n = 10)
  lt2 = as_lazy_tensor(ds2, list(x = c(NA, 5, 4)))
  lt2 = transform_lazy_tensor(lt2, po_add, c(NA, 5, 4))

  lt3 = c(lt[c(1, 2)], lt2[c(1, 2)])

  lt4 = transform_lazy_tensor(lt3, po_resize, c(NA, 20))


  expect_equal(address(lt4[[1L]]$data_descriptor), address(lt4[[2L]]$data_descriptor))
  expect_equal(address(lt4[[3L]]$data_descriptor), address(lt4[[4L]]$data_descriptor))
  expect_false(address(lt4[[1L]]$data_descriptor) == address(lt4[[3L]]$data_descriptor))

  expect_equal(length(lt4[[1L]]$data_descriptor$graph$pipeops), 2)
  expect_equal(length(lt4[[2L]]$data_descriptor$graph$pipeops), 2)

  expect_equal(lt4[[1L]]$data_descriptor$.pointer, c("resize", "output"))
  expect_equal(lt4[[3L]]$data_descriptor$.pointer, c("resize", "output"))

  expect_equal(length(lt4[[3L]]$data_descriptor$graph$pipeops), 3)
  expect_equal(length(lt4[[4L]]$data_descriptor$graph$pipeops), 3)
})

test_that("as_lazy_tensor works", {
  x = torch_randn(2, 3, 4)
  lt1 = as_lazy_tensor(x)
  expect_equal(lt1$shape, c(NA, 3, 4))
  expect_class(lt1, "lazy_tensor")
  expect_equal(length(lt1), 2)

  ds = random_dataset(5, 4)
  lt2 = as_lazy_tensor(ds, list(x = c(NA, 5, 4)))
  expect_class(lt2, "lazy_tensor")
  expect_equal(lt2$shape, c(NA, 5, 4))
  expect_equal(length(lt2), 10)
})
