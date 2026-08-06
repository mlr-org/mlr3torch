nn_scale = nn_module("nn_scale",
  initialize = function(n_features, init = 1) {
    self$weight = nn_parameter(torch_full(n_features, init))
  },
  forward = function(input) input * self$weight
)

test_that("pipeop_torch() generates a working PipeOpTorch", {
  Class = pipeop_torch("nn_scale", nn_scale,
    auxiliary = list(n_features = function(shapes_in, param_vals, task) tail(shapes_in[[1L]], 1L))
  )
  expect_class(Class, "R6ClassGenerator")
  expect_equal(Class$classname, "PipeOpTorchScale")

  obj = Class$new()
  expect_pipeop(obj)
  expect_class(obj, "PipeOpTorch")
  expect_equal(obj$id, "nn_scale")
  # the auxiliary argument is inferred from the shape, so it is not a hyperparameter
  expect_equal(obj$param_set$ids(), "init")
  expect_equal(obj$input$name, "input")
  expect_equal(obj$output$name, "output")
})

test_that("pipeop_torch() infers the output shapes by tracing the module", {
  obj = pipeop_torch("nn_scale", nn_scale,
    auxiliary = list(n_features = function(shapes_in, param_vals, task) tail(shapes_in[[1L]], 1L))
  )$new()
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 4L)))
  expect_equal(obj$shapes_out(list(c(8L, 4L))), list(output = c(8L, 4L)))

  # a module that changes the shape
  nn_first = nn_module("nn_first", forward = function(input) input[, 1L, drop = FALSE])
  expect_equal(pipeop_torch("nn_first", nn_first)$new()$shapes_out(list(c(NA, 4L))),
    list(output = c(NA, 1L)))
})

test_that("pipeop_torch() respects an explicit shapes_out", {
  obj = pipeop_torch("nn_scale", nn_scale,
    auxiliary = list(n_features = function(shapes_in, param_vals, task) tail(shapes_in[[1L]], 1L)),
    shapes_out = function(shapes_in, param_vals, task) list(c(shapes_in[[1L]][1L], 42L))
  )$new()
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 42L)))

  # NULL means the operator does not change the shape
  nop = pipeop_torch("nn_scale", nn_scale, shapes_out = NULL,
    auxiliary = list(n_features = function(shapes_in, param_vals, task) tail(shapes_in[[1L]], 1L))
  )$new()
  expect_equal(nop$shapes_out(list(c(NA, 4L))), list(output = c(NA, 4L)))
})

test_that("pipeop_torch() works with multiple input and output channels", {
  nn_custom = nn_module("nn_custom",
    initialize = function(d_in1, d_in2, d_out1, d_out2, bias = TRUE) {
      self$linear1 = nn_linear(d_in1, d_out1, bias)
      self$linear2 = nn_linear(d_in2, d_out2, bias)
    },
    forward = function(input1, input2) {
      list(output1 = self$linear1(input1), output2 = self$linear2(input2))
    }
  )
  obj = pipeop_torch("nn_custom", nn_custom, outname = c("output1", "output2"),
    auxiliary = list(
      d_in1 = function(shapes_in, param_vals, task) tail(shapes_in$input1, 1L),
      d_in2 = function(shapes_in, param_vals, task) tail(shapes_in$input2, 1L)
    )
  )$new(param_vals = list(d_out1 = 10L, d_out2 = 20L))

  # the input channels are the arguments of $forward()
  expect_equal(obj$input$name, c("input1", "input2"))
  expect_equal(obj$output$name, c("output1", "output2"))
  expect_equal(obj$shapes_out(list(input1 = c(NA, 2L), input2 = c(NA, 3L))),
    list(output1 = c(NA, 10L), output2 = c(NA, 20L)))
})

test_that("shape_dependent_params can be given instead of auxiliary", {
  obj = pipeop_torch("nn_scale", nn_scale,
    param_set = ps(init = p_dbl(tags = "train")),
    shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals, list(n_features = tail(shapes_in[[1L]], 1L)))
    }
  )$new(param_vals = list(init = 2))
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 4L)))

  md = po("torch_ingress_num")$train(list(tsk("iris")))[[1L]]
  module = obj$train(list(md))[[1L]]$graph$pipeops$nn_scale$module
  expect_equal(as.numeric(module$weight), rep(2, 4L))
})

test_that("a PipeOp from pipeop_torch() can be used in a network", {
  obj = pipeop_torch("nn_scale", nn_scale,
    auxiliary = list(n_features = function(shapes_in, param_vals, task) tail(shapes_in[[1L]], 1L))
  )$new()
  graph = po("torch_ingress_num") %>>% obj %>>% po("nn_head")
  md = graph$train(tsk("iris"))[[1L]]
  network = model_descriptor_to_module(md)
  expect_class(network, "nn_graph")

  x = torch_randn(2L, 4L)
  expect_equal(dim(with_no_grad(network(torch_ingress_num.input = x))), c(2L, 3L))

  # the module was built with the inferred number of features
  expect_equal(as.numeric(md$graph$pipeops$nn_scale$module$weight), rep(1, 4L))
})

test_that("pipeop_torch() argument checks", {
  aux = list(n_features = function(shapes_in, param_vals, task) tail(shapes_in[[1L]], 1L))
  expect_error(pipeop_torch("nn_scale", nn_scale, auxiliary = aux,
    shape_dependent_params = function(shapes_in, param_vals, task) param_vals),
    "not both", fixed = TRUE)
  expect_error(pipeop_torch("nn_scale", nn_scale, auxiliary = list(nonexistent = function(...) 1)),
    "auxiliary")
  expect_error(pipeop_torch("nn_scale", nn_scale, auxiliary = aux,
    param_set = ps(n_features = p_int(tags = "train"))), "disjunct")
  expect_error(pipeop_torch("nn_scale", "not a module"), "nn_module_generator")
})

test_that("as_pipeop() works for nn_module_generators", {
  nn_square = nn_module("nn_square", forward = function(input) input^2)
  obj = as_pipeop(nn_square)
  expect_pipeop(obj)
  expect_class(obj, "PipeOpTorch")
  expect_equal(obj$id, "nn_square")
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 4L)))

  graph = po("torch_ingress_num") %>>% obj %>>% po("nn_head")
  network = model_descriptor_to_module(graph$train(tsk("iris"))[[1L]])
  x = torch_randn(2L, 4L)
  expect_equal(dim(with_no_grad(network(torch_ingress_num.input = x))), c(2L, 3L))
})
