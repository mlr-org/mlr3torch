test_that("PipeOpModule can combine vararg channel and non-vararg channel", {

  module = nn_module("nn_test",
    forward = function(..., input) {
      input * torch_sum(torch_stack(...), dim = 1L)
    }
  )()

  obj = PipeOpModule$new(module = module, inname = c("x1", "x2", "input"), outname = "output", id = "test")

  x = list(x1 = torch_tensor(1), x2 = torch_tensor(2), input = torch_tensor(3))
  obj$train(input = x)


})
