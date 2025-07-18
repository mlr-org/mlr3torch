test_that("nn works", {
  x = nn("linear", out_features = 3)
  expect_equal(x$id, "linear")
  expect_class(x, "PipeOpTorchLinear")
  expect_equal(x$param_set$values$out_features, 3)
})

test_that("works with unnamed arguments", {
  obj = nn("block", nn("linear"))
  expect_equal(obj$id, "block")
  expect_equal(obj$block$pipeops[[1]]$id, "linear")
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
