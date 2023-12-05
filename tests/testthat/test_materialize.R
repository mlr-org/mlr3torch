test_that("materialize works on lazy_tensor", {
  ds = random_dataset(5, 4, n = 10)
  lt = as_lazy_tensor(ds, list(x = c(NA, 5, 4)))

  output = materialize(lt, device = "cpu", rbind = TRUE)
  expect_class(output, "torch_tensor")
  expect_equal(output$shape, c(10, 5, 4))
  expect_true(output$device == torch_device("cpu"))
  expect_true(torch_equal(output, ds$x))
  # the correct elements are returned
  expect_torch_equal(ds$.getbatch(1)[[1]], materialize(lt[1])[[1]])
  expect_torch_equal(ds$.getbatch(2)[[1]], materialize(lt[2])[[1]])
  expect_torch_equal(ds$.getbatch(2:1)[[1]], materialize(lt[2:1], rbind = TRUE))

  output_meta_list = materialize(lt, device = "meta", rbind = FALSE)
  output_meta_tnsr = materialize(lt, device = "meta", rbind = TRUE)

  expect_equal(torch_cat(output_meta_list, dim = 1L)$shape, output_meta_tnsr$shape)
  expect_true(output_meta_tnsr$device == torch_device("meta"))

  expect_error(materialize(lazy_tensor()), "Cannot materialize ")
})

test_that("materialize works with differing shapes (hence uses .getitem)", {
  task = nano_dogs_vs_cats()

  lt = task$data(1:2, cols = "x")[[1L]]

  res1 = materialize(lt, rbind = FALSE, device = "meta")
  expect_list(res1, types = "torch_tensor")
  expect_false(identical(res1[[1]]$shape, res1[[2]]$shape))
  expect_true(res1[[1]]$device == torch_device("meta"))

  # cannot rbind tensors with varying shapes
  expect_error(materialize(lt, rbind = TRUE))
})

test_that("materialize works with same shapes and .getbatch method", {
  task = tsk("lazy_iris")

  x = task$data(1:2, cols = "x")[[1L]]

  res1 = materialize(x, rbind = FALSE, device = "meta")
  expect_list(res1, types = "torch_tensor")
  expect_true(res1[[1]]$device == torch_device("meta"))

  res2 = materialize(x, rbind = TRUE, device = "meta")
  expect_class(res2, "torch_tensor")
  expect_true(res2$device == torch_device("meta"))

  res1cpu = materialize(x, rbind = FALSE)
  res2cpu = materialize(x, rbind = TRUE)
  expect_torch_equal(torch_cat(res1cpu, dim = 1L), res2cpu)
  expect_equal(res2cpu$shape, res2$shape)
})

test_that("materialize works with same shapes and .getitem method", {
  task = nano_mnist()

  x = task$data(1:2, cols = "image")[[1L]]

  res1 = materialize(x, rbind = FALSE, device = "meta")
  expect_list(res1, types = "torch_tensor")
  expect_true(res1[[1]]$device == torch_device("meta"))

  res2 = materialize(x, rbind = TRUE, device = "meta")
  expect_class(res2, "torch_tensor")
  expect_true(res2$device == torch_device("meta"))
  expect_equal(res2$shape, c(2, 1, 28, 28))

  res1cpu = materialize(x, rbind = FALSE)
  res2cpu = materialize(x, rbind = TRUE)
  expect_torch_equal(torch_cat(res1cpu, dim = 1L), res2cpu)
  expect_equal(res2cpu$shape, res2$shape)
})

test_that("materialize_internal works", {
  expect_error(materialize_internal(lazy_tensor()), "Cannot materialize ")
  task = tsk("lazy_iris")
  x = task$data(1:2, cols = "x")[[1L]]
  res1 = materialize(x)
  res2 = materialize(x, rbind = TRUE)
  expect_list(res1, types = "torch_tensor")
  expect_class(res2, "torch_tensor")
  expect_torch_equal(torch_cat(res1, dim = 1L), res2)

  res1cpu = materialize(x, rbind = FALSE)
  res2cpu = materialize(x, rbind = TRUE)
  expect_torch_equal(torch_cat(res1cpu, dim = 1L), res2cpu)
})


test_that("materialize.list works", {
  df = nano_mnist()$data(1:10, cols = "image")

  out = materialize(df, rbind = TRUE)
  expect_list(out)
  expect_equal(names(out), "image")
  expect_class(out$image, "torch_tensor")
  expect_equal(out$image$shape, c(10, 1, 28, 28))

  # to check:
  # a) caching works when en / disabling cache manually
  # b) set_keep_results stuff works
  # c) default = "auto" works as expected

  # TODO

})

test_that("materialize_internal: caching of graphs works", {
  cache = new.env()
  task = tsk("lazy_iris")
  # need to rename because of a nasty bug in pipelines:
  # https://github.com/mlr-org/mlr3pipelines/issues/738`

  dt = task$data(1:2, cols = "x")
  dt$x1 = dt$x
  names(dt) = c("x1", "x2")

  materialize(dt, rbind = TRUE)

})

test_that("materialize_internal: caching of datasets works", {
  ds = dataset(
    initialize = function() {
      self$x = torch_randn(10, 3)
      self$count = 0
    },
    .getitem = function(i) {
      self$count = self$count + 1
      list(x = self$x[i, ])
    },
    .length = function() {
      10
    }
  )()
  x1 = as_lazy_tensor(ds, list(x = c(NA, 3)))
  x2 = as_lazy_tensor(ds, list(x = c(NA, 3)))

  # hashes of environments change after a function was called (?)
  # https://github.com/mlr-org/mlr3torch/issues/156
  expect_equal(
    x1$.dataset_hash,
    x2$.dataset_hash
  )

  dd1 = DataDescriptor(ds, list(x = c(NA, 3)))
  dd2 = DataDescriptor(ds, list(x = c(NA, 3)))

  dd1$.dataset_hash
  dd2$.dataset_hash


  d = data.table(x1 = x1, x2 = x2)
  materialize(d, rbind = TRUE, cache = new.env())
  expect_true(ds$count == 10)
})

test_that("materialize_internal: resets everything to previous state", {


})

test_that("materialize_internal: set_keep_results works", {

})

test_that("PipeOpFeatureUnion can properly check whether two lazy tensors are identical", {
  # when lazy_tensor only stored the integers in the vec_data() (and not integer + hash) this test failed
  task = tsk("lazy_iris")

  graph = po("nop") %>>%
    list(po("preproc_torch", function(x) x + 1), po("trafo_nop")) %>>%
    po("featureunion")

  expect_error(graph$train(task), "cannot aggregate different features sharing")
})


