test_that("PipeOpModule works", {
  po_module = PipeOpModule$new("linear", torch::nn_linear(10, 20), inname = "abc", outname = "xyz")
  x = torch::torch_randn(16, 10)
  # This calls the forward function of the wrapped module.
  y_po = with_no_grad(po_module$train(list(input = x)))
  y = with_no_grad(po_module$module(x))
  expect_list(y_po, types = "torch_tensor")
  expect_true(names(y_po) == "xyz")
  expect_true(torch_equal(y, y_po[[1L]]))

  # multiple input and output channels
  nn_custom = torch::nn_module("nn_custom",
    initialize = function(in_features, out_features) {
      self$lin1 = torch::nn_linear(in_features, out_features)
      self$lin2 = torch::nn_linear(in_features, out_features)
    },
    forward = function(x, z) {
      list(out1 = self$lin1(x), out2 = torch::nnf_relu(self$lin2(z)))
    }
  )
  d = 3

  module = nn_custom(d, 2)
  po_module = PipeOpModule$new("custom", module, inname = c("x", "z"), outname = c("out1", "out2"))
  input = list(x = torch_randn(2, d), z = torch_randn(2, d))
  y = po_module$train(input)
  expect_true(all(sort(names(y)) == c("out1", "out2")))
  expect_list(y, types = "torch_tensor")
})
