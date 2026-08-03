test_that("nn works", {
  x = nn("linear", out_features = 3)
  expect_equal(x$id, "linear")
  expect_class(x, "PipeOpTorchLinear")
  expect_equal(x$param_set$values$out_features, 3)
})

test_that("overwrite id", {
  obj = nn("linear", id = "abc")
  expect_equal(obj$id, "abc")
})

test_that("unnamed arg", {
  graph = po("torch_ingress_num") %>>% nn("block", nn("linear", out_features = 3), n_blocks = 2)
  md = graph$train(tsk("iris"))[[1L]]
  network = model_descriptor_to_module(md)
  expect_equal(network$module_list[[1]]$out_features, 3)
  expect_equal(network$module_list[[2]]$out_features, 3)
})

test_that("nn works with cov", {
  expect_equal(nn("conv3d")$id, "conv3d")
})

test_that("numeric suffix disambiguates repeated layers", {
  x = nn("linear_1", out_features = 3)
  expect_class(x, "PipeOpTorchLinear")
  expect_equal(x$id, "linear_1")
  expect_equal(x$param_set$values$out_features, 3)

  expect_equal(nn("relu_2")$id, "relu_2")

  # keys that end in a digit without an underscore are not affected
  expect_equal(nn("max_pool1d")$id, "max_pool1d")
  expect_equal(nn("relu6")$id, "relu6")

  # an explicit id still wins
  expect_equal(nn("linear_1", id = "abc")$id, "abc")

  graph = po("torch_ingress_num") %>>%
    nn("linear_1", out_features = 3) %>>%
    nn("relu_1") %>>%
    nn("linear_2", out_features = 4)
  expect_equal(graph$ids(), c("torch_ingress_num", "linear_1", "relu_1", "linear_2"))
})
