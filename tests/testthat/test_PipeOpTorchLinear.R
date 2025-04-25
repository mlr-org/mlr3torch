test_that("PipeOpTorchLinear works", {
  po_linear = po("nn_linear", out_features = 10)
  graph = po("torch_ingress_num") %>>% po_linear
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_linear", task, "nn_linear")
})

test_that("PipeOpTorchLinear paramtest", {
  po_linear = po("nn_linear", out_features = 10)
  res = expect_paramset(po_linear, nn_linear, exclude = "in_features")
  expect_paramtest(res)
})

test_that("NA in second dimension", {
  ds = dataset(
    initialize = function() {
      self$xs = lapply(1:10, function(i) torch_randn(sample(1:10, 1), 10))
    },
    .getitem = function(i) {
      list(x = self$xs[[i]])
    },
    .length = function() {
      length(self$xs)
    }
  )()

  task = as_task_regr(data.table(
    x = as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, NA, NA))),
    y = rnorm(10)
  ), target = "y", id = "test")

  graph = po("torch_ingress_ltnsr") %>>% po("nn_linear", out_features = 10)

  expect_error(graph$train(task), "Please provide a known shape")

  task = as_task_regr(data.table(
    x = as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, NA, 10))),
    y = rnorm(10)
  ), target = "y", id = "test")

  md = graph$train(task)[[1L]]
  expect_equal(md$pointer_shape, c(NA, NA, 10))
  net = model_descriptor_to_module(md)
  expect_equal(net(torch_randn(1, 2, 10))$shape, c(1, 2, 10))
})
