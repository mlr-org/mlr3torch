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
  expect_error(materialize_internal(lazy_tensor()), "Cannot access data descriptor")
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
  materialize(d, rbind = TRUE, cache = hashtab())
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

test_that("0-length", {
  expect_equal(torch_empty(0L), materialize(lazy_tensor(), rbind = TRUE))
  expect_equal(list(), materialize(lazy_tensor(), rbind = FALSE))
})

test_that("materialize.data.frame with empty data.frame", {
  lt = as_lazy_tensor(torch_randn(5, 3))
  df = data.frame(x = I(lt[integer(0)]))

  res_rbind = materialize(df, rbind = TRUE)
  expect_list(res_rbind)
  expect_equal(names(res_rbind), "x")
  expect_class(res_rbind$x, "torch_tensor")
  expect_equal(res_rbind$x$shape, 0L)

  res_list = materialize(df, rbind = FALSE)
  expect_list(res_list)
  expect_equal(names(res_list), "x")
  expect_list(res_list$x, len = 0)
})

test_that("materialize with shape (NA, NA) and .getbatch implementation", {
  # this can e.g. happen when we do padding in the dataset
  ds = dataset(
    initialize = function() {
      self$x = torch_randn(10, 3)
    },
    .getbatch = function(i) {
      list(x = self$x[i, , drop = FALSE])
    },
    .length = function() 10
  )()
  lt = as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, NA)))
  expect_class(materialize(lt, rbind = TRUE), "torch_tensor")

  mod = nn_module(
    forward = function(x) {
      torch_reshape(x, c(-1, 3, 1))
    }
  )()
  po_module = po("module", module = mod, id = "mod")
  lt1 = transform_lazy_tensor(lt, po_module, shape = c(NA, NA, 1))
  expect_equal(materialize(lt1[1])[[1L]]$shape, c(3, 1))
})

test_that("materialize()'s cache is keyed by the objects that identify an entry", {
  ds = dataset(
    initialize = function() self$x = torch_randn(10, 3),
    .getbatch = function(i) list(x = self$x[i, , drop = FALSE]),
    .length = function() 10
  )()
  x1 = as_lazy_tensor(ds, list(x = c(NA, 3)))
  x2 = as_lazy_tensor(ds, list(x = c(NA, 3)))
  d = data.table(x1 = x1, x2 = x2)

  # the cache used to be an `environment()`, whose keys are strings and therefore had to be digested
  expect_error(materialize(d, rbind = TRUE, cache = new.env()), "hashtab")

  cache = hashtab()
  expect_equal(materialize(d, rbind = TRUE, cache = cache), materialize(d, rbind = TRUE, cache = NULL))

  # keys are the identifying objects themselves, so `identical()` decides a hit and two distinct
  # keys cannot end up sharing an entry through a digest collision
  keys = list()
  collect_key = function(key, value) {
    keys[[length(keys) + 1L]] <<- key
  }
  maphash(cache, collect_key)
  expect_true(length(keys) > 0L)
  expect_true(every(keys, is.list))
  # the dataset-level and the graph-level entries are kept apart by a leading tag
  expect_set_equal(unique(map_chr(keys, function(key) key[[1L]])), c("input", "output"))

  # the dataset is in the key as the object, not as `dd()$dataset_hash`, which is only
  # `calculate_hash(address(dataset))` and could therefore collide for two distinct datasets
  expect_true(every(keys, function(key) some(key, function(part) identical(part, ds))))
  expect_true(every(keys, function(key) !any(map_lgl(key, identical, dd(x1)$dataset_hash))))

  # a second dataset that is equal in every respect still gets its own entries
  ds2 = dataset(
    initialize = function() self$x = torch_randn(10, 3),
    .getbatch = function(i) list(x = self$x[i, , drop = FALSE]),
    .length = function() 10
  )()
  y = as_lazy_tensor(ds2, list(x = c(NA, 3)))
  before = numhash(cache)
  materialize(data.table(y1 = y, y2 = y), rbind = TRUE, cache = cache)
  expect_true(numhash(cache) > before)
})

test_that("materialize()'s cache keeps entries for different graphs apart", {
  # two lazy tensor columns over the same dataset that differ only in their preprocessing graph:
  # they share the cached dataset output, but must not share the cached graph output
  x = as_lazy_tensor(as.double(1:10))
  task = as_task_regr(data.table(y = 1:10, a = x, b = x), target = "y")
  task = po("preproc_torch", fn = mlr3misc::crate(function(x) x + 1),
    stages = "both", affect_columns = selector_name("a"))$train(list(task))[[1L]]
  task = po("preproc_torch", fn = mlr3misc::crate(function(x) x * 2),
    stages = "both", affect_columns = selector_name("b"))$train(list(task))[[1L]]

  d = task$data(cols = c("a", "b"))
  cached = materialize(d, rbind = TRUE, cache = hashtab())
  expect_equal(cached, materialize(d, rbind = TRUE, cache = NULL))
  expect_equal(as.numeric(cached$a), as.double(1:10) + 1)
  expect_equal(as.numeric(cached$b), as.double(1:10) * 2)
})

test_that("materialize()'s cache runs a merged graph once for all of its columns", {
  # `merge_compatible_lazy_tensor_graphs()` merges the columns that share a dataset into one graph
  # and hands each of them a `DataDescriptor` over that same graph object, differing only in
  # `pointer`. That is what the output-level cache is for, and why the graph is in the key as the
  # object rather than as `$hash`.
  ds = dataset(
    initialize = function() self$x = torch_randn(10, 3),
    .getbatch = function(i) list(x = self$x[i, , drop = FALSE]),
    .length = function() 10
  )()
  x = as_lazy_tensor(ds, list(x = c(NA, 3)))
  task = as_task_regr(data.table(y = 1:10, a = x, b = x), target = "y")
  task = po("preproc_torch", fn = mlr3misc::crate(function(x) x + 1), stages = "both",
    affect_columns = selector_name("a"))$train(list(task))[[1L]]
  task = po("preproc_torch", fn = mlr3misc::crate(function(x) x * 2), stages = "both",
    affect_columns = selector_name("b"))$train(list(task))[[1L]]

  merged = merge_lazy_tensor_graphs(task$data(cols = c("a", "b")))
  expect_true(identical(dd(merged$a)$graph, dd(merged$b)$graph))
  expect_equal(dd(merged$a)$hash, dd(merged$b)$hash)

  cache = hashtab()
  cached = materialize(merged, rbind = TRUE, cache = cache)
  expect_equal(cached, materialize(merged, rbind = TRUE, cache = NULL))
  raw = materialize(x, rbind = TRUE)
  expect_true(torch_allclose(cached$a, raw + 1))
  expect_true(torch_allclose(cached$b, raw * 2))

  # one dataset read and one graph run for the two columns, rather than one of each per column
  tags = character()
  collect_tag = function(key, value) {
    tags <<- c(tags, key[[1L]])
  }
  maphash(cache, collect_tag)
  expect_equal(sort(tags), c("input", "output"))
})
