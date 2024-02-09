test_that("materialize works on lazy_tensor", {
  ds = random_dataset(5, 4, n = 10)
  lt = as_lazy_tensor(ds, list(x = c(NA, 5, 4)))

  output = materialize(lt, device = "cpu", rbind = TRUE)
  expect_class(output, "torch_tensor")
  expect_equal(output$shape, c(10, 5, 4))
  expect_true(output$device == torch_device("cpu"))
  expect_true(torch_equal(output, ds$x))
  # the correct elements are returned
  expect_equal(ds$.getbatch(1)[[1]], materialize(lt[1])[[1]]$unsqueeze(1))
  expect_equal(ds$.getbatch(2)[[1]], materialize(lt[2])[[1]]$unsqueeze(1))
  expect_equal(ds$.getbatch(2:1)[[1]], materialize(lt[2:1], rbind = TRUE))

  output_meta_list = materialize(lt, device = "meta", rbind = FALSE)
  output_meta_tnsr = materialize(lt, device = "meta", rbind = TRUE)

  expect_equal(torch_cat(map(output_meta_list, function(x) x$unsqueeze(1)), dim = 1L)$shape, output_meta_tnsr$shape)
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
  expect_equal(list_to_batch(res1cpu), res2cpu)
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
  expect_equal(list_to_batch(res1cpu), res2cpu)
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
  expect_equal(list_to_batch(res1), res2)

  res1cpu = materialize(x, rbind = FALSE)
  res2cpu = materialize(x, rbind = TRUE)
  expect_equal(list_to_batch(res1cpu), res2cpu)
})


test_that("materialize.list works", {
  df = nano_mnist()$data(1:10, cols = "image")

  out = materialize(df, rbind = TRUE)
  expect_list(out)
  expect_equal(names(out), "image")
  expect_class(out$image, "torch_tensor")
  expect_equal(out$image$shape, c(10, 1, 28, 28))
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
    dd(x1)$dataset_hash,
    dd(x2)$dataset_hash
  )

  dd1 = DataDescriptor$new(ds, list(x = c(NA, 3)))
  dd2 = DataDescriptor$new(ds, list(x = c(NA, 3)))

  dd1$dataset_hash
  dd2$dataset_hash

  # need to do this, because DataDescritor creation retrieves an example batch to verify the shapes.
  ds$count = 0

  d = data.table(x1 = x1, x2 = x2)
  materialize(d, rbind = TRUE, cache = new.env())
  expect_true(ds$count == 10)
})

test_that("materialize with varying shapes", {
  task = nano_dogs_vs_cats()$filter(1:2)
  x = materialize(task$data()$x, rbind = FALSE)
  expect_list(x, types = "torch_tensor")
  expect_equal(x[[1]]$shape[1L], 3)
  expect_equal(x[[2]]$shape[1L], 3)

  # shapes don't fit together
  expect_error(materialize(task$data()$x, rbind = TRUE))

  e = new.env()
  e$a = 2L

  # depending on whether we apply this per row or per batch, we will get different results
  # (second's sum(abs()) is either zero or non-zero)
  fn = crate(function(x) {
    a <<- a - 1
    x * a
  }, .parent = e)
  po_test = pipeop_preproc_torch("trafo_test", fn = fn)$new()
  # is processed batch-wise ->
  task2 = po_test$train(list(nano_mnist()$filter(1:2)))[[1L]]

  x2 = materialize(task2$data()$image, rbind = TRUE)
  expect_true(as.logical(sum(abs(x2[2, ..])) != 0))

  e$a = 2
  x2 = materialize(task2$data()$image, rbind = FALSE)
  expect_true(as.logical(sum(abs(x2[[2L]])) != 0))

  e$a = 2
  x3 = materialize(po_test$train(list(task))[[1L]]$data()$x)
  expect_true(as.logical(sum(abs(x3[[2L]])) == 0L))
})

test_that("PipeOpFeatureUnion can properly check whether two lazy tensors are identical", {
  # when lazy_tensor only stored the integers in the vec_data() (and not integer + hash) this test failed
  task = tsk("lazy_iris")

  graph = po("nop") %>>%
    list(po("preproc_torch", function(x) x + 1, stages_init = "both"), po("trafo_nop")) %>>%
    po("featureunion")

  expect_error(graph$train(task), "cannot aggregate different features sharing")
})
