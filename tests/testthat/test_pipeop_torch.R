scale_args = list(
  initialize = function(shapes_in, init = 1) {
    self$weight = nn_parameter(torch_full(tail(shapes_in[[1L]], 1L), init))
  },
  forward = function(input) input * self$weight,
  shapes_out = function(shapes_in) shapes_in
)

pipeop_torch_scale = function(...) {
  invoke(pipeop_torch, "nn_scale", .args = insert_named(scale_args, list(...)))
}

test_that("pipeop_torch() generates a working PipeOpTorch", {
  Class = pipeop_torch_scale()
  expect_class(Class, "R6ClassGenerator")
  expect_equal(Class$classname, "PipeOpTorchScale")

  obj = Class$new()
  expect_pipeop(obj)
  expect_class(obj, "PipeOpTorch")
  expect_equal(obj$id, "nn_scale")
  # `shapes_in` is supplied by the PipeOp, so it is not a hyperparameter
  expect_equal(obj$param_set$ids(), "init")
  expect_equal(obj$input$name, "input")
  expect_equal(obj$output$name, "output")
})

test_that("the module gets the shapes and the task it asks for", {
  md = po("torch_ingress_num")$train(list(tsk("iris")))[[1L]]

  obj = pipeop_torch_scale()$new()
  module = obj$train(list(md))[[1L]]$graph$pipeops$nn_scale$module
  # the module was built with the number of features taken from the input shape
  expect_equal(as.numeric(module$weight), rep(1, 4L))

  obj = pipeop_torch("nn_peek",
    initialize = function(shapes_in, task) {
      self$seen_shapes = shapes_in
      self$seen_task = task$id
    },
    forward = function(input) input,
    shapes_out = function(shapes_in) shapes_in
  )$new()
  module = obj$train(list(md))[[1L]]$graph$pipeops$nn_peek$module
  expect_equal(module$seen_shapes, list(input = c(NA, 4L)))
  expect_equal(module$seen_task, "iris")

  # a module that asks for neither is written as it would be outside of mlr3torch
  obj = pipeop_torch("nn_square", forward = function(input) input^2,
    shapes_out = function(shapes_in) shapes_in)$new()
  expect_equal(obj$param_set$ids(), character(0))
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 4L)))
})

test_that("pipeop_torch() infers the parameter set from the constructor", {
  obj = pipeop_torch("nn_custom",
    initialize = function(shapes_in, task, required_arg, optional_arg = 1) NULL,
    forward = function(input) input,
    shapes_out = function(shapes_in) shapes_in
  )$new()
  expect_equal(obj$param_set$ids(), c("required_arg", "optional_arg"))
  expect_true(all(map_lgl(obj$param_set$ids(), function(id) obj$param_set$class[[id]] == "ParamUty")))
  # an argument without a default cannot be left unset
  expect_true("required" %in% obj$param_set$tags$required_arg)
  expect_false("required" %in% obj$param_set$tags$optional_arg)
  expect_error(obj$shapes_out(list(c(NA, 4L))), "required_arg")
})

test_that("shapes_out only gets the arguments it declares", {
  obj = pipeop_torch_scale(shapes_out = function(shapes_in) list(c(shapes_in[[1L]][1L], 42L)))$new()
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 42L)))

  obj = pipeop_torch("nn_custom",
    initialize = function(out_features) NULL,
    forward = function(input) input,
    shapes_out = function(shapes_in, param_vals, task) {
      list(c(head(shapes_in[[1L]], -1L), param_vals$out_features, length(task$class_names)))
    }
  )$new(param_vals = list(out_features = 5L))
  expect_equal(obj$shapes_out(list(c(NA, 4L)), task = tsk("iris")), list(output = c(NA, 5L, 3L)))
})

test_that("an existing module is wrapped by calling it from initialize()", {
  obj = pipeop_torch("nn_linear2",
    initialize = function(shapes_in, out_features, bias = TRUE) {
      self$linear = nn_linear(tail(shapes_in[[1L]], 1L), out_features, bias)
    },
    forward = function(input) self$linear(input),
    shapes_out = function(shapes_in, param_vals) {
      list(c(head(shapes_in[[1L]], -1L), param_vals$out_features))
    }
  )$new(param_vals = list(out_features = 10L))

  expect_equal(obj$param_set$ids(), c("out_features", "bias"))
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 10L)))

  md = po("torch_ingress_num")$train(list(tsk("iris")))[[1L]]
  module = obj$train(list(md))[[1L]]$graph$pipeops$nn_linear2$module
  expect_equal(dim(module$linear$weight), c(10L, 4L))
})

test_that("pipeop_torch() works with multiple input and output channels", {
  obj = pipeop_torch("nn_custom",
    initialize = function(shapes_in, d_out1, d_out2, bias = TRUE) {
      self$linear1 = nn_linear(tail(shapes_in$input1, 1L), d_out1, bias)
      self$linear2 = nn_linear(tail(shapes_in$input2, 1L), d_out2, bias)
    },
    forward = function(input1, input2) {
      list(self$linear1(input1), self$linear2(input2))
    },
    out_channels = 2L,
    shapes_out = function(shapes_in, param_vals) {
      list(c(head(shapes_in$input1, -1L), param_vals$d_out1),
        c(head(shapes_in$input2, -1L), param_vals$d_out2))
    }
  )$new(param_vals = list(d_out1 = 10L, d_out2 = 20L))

  # the input channels are the arguments of $forward()
  expect_equal(obj$input$name, c("input1", "input2"))
  expect_equal(obj$output$name, c("output1", "output2"))
  expect_equal(obj$shapes_out(list(input1 = c(NA, 2L), input2 = c(NA, 3L))),
    list(output1 = c(NA, 10L), output2 = c(NA, 20L)))
})

test_that("channels can be given as names or as a count", {
  make = function(...) {
    invoke(pipeop_torch, "nn_id", forward = function(...) ..1,
      shapes_out = function(shapes_in) shapes_in[1L], .args = list(...))$new()
  }
  # a `forward()` that only takes `...` becomes a vararg channel, as does `in_channels = 0`
  expect_equal(make()$input$name, "...")
  expect_equal(make(in_channels = 0L)$input$name, "...")
  expect_equal(make(in_channels = 1L)$input$name, "input")
  expect_equal(make(in_channels = 3L)$input$name, c("input1", "input2", "input3"))
  expect_equal(make(in_channels = c("a", "b"))$input$name, c("a", "b"))
  expect_equal(make(out_channels = 2L)$output$name, c("output1", "output2"))
  expect_equal(make(out_channels = "prediction")$output$name, "prediction")
  expect_error(make(out_channels = 0L), "number of output channels")
})

test_that("a PipeOp from pipeop_torch() can be used in a network", {
  obj = pipeop_torch_scale()$new()
  graph = po("torch_ingress_num") %>>% obj %>>% po("nn_head")
  md = graph$train(tsk("iris"))[[1L]]
  network = model_descriptor_to_module(md)
  expect_class(network, "nn_graph")

  x = torch_randn(2L, 4L)
  expect_equal(dim(with_no_grad(network(torch_ingress_num.input = x))), c(2L, 3L))
})

test_that("the functions keep the environments they were written in", {
  # R6 rebinds the environment of anything it is given as a method, which would silently drop the
  # closures of the module's methods and of `shapes_out()`
  make = function(k) {
    pipeop_torch("nn_take",
      forward = function(input) input[, 1:k],
      shapes_out = function(shapes_in) list(c(shapes_in[[1L]][1L], k))
    )
  }
  obj = make(2L)$new()
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 2L)))
  network = model_descriptor_to_module((po("torch_ingress_num") %>>% obj)$train(tsk("iris"))[[1L]])
  expect_equal(dim(with_no_grad(network(torch_ingress_num.input = torch_randn(3L, 4L)))), c(3L, 2L))

  # ... also when they are written somewhere other than the call site
  shapes_out = local({
    k = 2L
    function(shapes_in) list(c(shapes_in[[1L]][1L], k))
  })
  obj = pipeop_torch("nn_x", forward = function(input) input, shapes_out = shapes_out)$new()
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 2L)))
})

test_that("the generated class does not depend on where it was created", {
  # package internals must resolve in mlr3torch's namespace, not on the caller's search path
  env = new.env(parent = baseenv())
  env$pipeop_torch = pipeop_torch
  obj = eval(quote(pipeop_torch("nn_x", forward = function(input) input,
    shapes_out = function(shapes_in) shapes_in)$new()), env)
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 4L)))
})

test_that("shapes_out() has to return one shape per output channel", {
  obj = pipeop_torch("nn_a", forward = function(input) input,
    shapes_out = function(shapes_in) c(NA, 7L))$new()
  expect_error(obj$shapes_out(list(c(NA, 4L))), "must return a list of 1 shape")

  obj = pipeop_torch("nn_b", forward = function(input) input, out_channels = 2L,
    shapes_out = function(shapes_in) shapes_in)$new()
  expect_error(obj$shapes_out(list(c(NA, 4L))), "must return a list of 2 shape")
})

test_that("operators that differ in what they do differ in their phash", {
  C1 = pipeop_torch("nn_m", forward = function(input) input,
    shapes_out = function(shapes_in) shapes_in)
  C2 = pipeop_torch("nn_m", forward = function(input) input * 2,
    shapes_out = function(shapes_in) list(c(shapes_in[[1L]][1L], 99L)))
  expect_false(C1$new()$phash == C2$new()$phash)
  expect_equal(C1$new()$phash, C1$new()$phash)

  # and the ParamSet is not shared between instances
  obj1 = pipeop_torch_scale()$new()
  obj2 = pipeop_torch_scale()$new()
  obj1$param_set$set_values(init = 2)
  expect_equal(obj2$param_set$values, named_list())
})

test_that("pipeop_torch() argument checks", {
  expect_error(pipeop_torch("nn_scale", forward = function(input) input), "shapes_out")
  expect_error(pipeop_torch("nn_scale", shapes_out = function(shapes_in) shapes_in), "forward")
  expect_error(pipeop_torch_scale(param_set = ps(shapes_in = p_uty(tags = "train"))), "disjunct")
  expect_error(pipeop_torch_scale(param_set = ps(nonexistent = p_uty(tags = "train"))),
    "parameter ids")
})

test_that("as_pipeop() converts a module that describes itself", {
  nn_scale = invoke(nn_module, "nn_scale", .args = scale_args)
  obj = as_pipeop(nn_scale)
  expect_pipeop(obj)
  expect_class(obj, "PipeOpTorch")
  expect_equal(obj$id, "nn_scale")
  expect_equal(obj$param_set$ids(), "init")
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 4L)))

  graph = po("torch_ingress_num") %>>% obj %>>% po("nn_head")
  network = model_descriptor_to_module(graph$train(tsk("iris"))[[1L]])
  x = torch_randn(2L, 4L)
  expect_equal(dim(with_no_grad(network(torch_ingress_num.input = x))), c(2L, 3L))

  # without a `shapes_out()` method there is nothing to convert
  nn_square = nn_module("nn_square", forward = function(input) input^2)
  expect_error(as_pipeop(nn_square), "no `shapes_out()` method", fixed = TRUE)

  # the module's methods keep their environments, wherever it was written
  make_module = function(k) {
    nn_module("nn_take",
      forward = function(input) input[, 1:k],
      shapes_out = function(shapes_in) list(c(shapes_in[[1L]][1L], k))
    )
  }
  expect_equal(as_pipeop(make_module(2L))$shapes_out(list(c(NA, 5L))), list(output = c(NA, 2L)))

  # `shapes_out()` is called before any module exists, so `self` is not available to it
  nn_selfish = nn_module("nn_selfish", forward = function(input) input,
    shapes_out = function(shapes_in) self$anything)
  expect_error(as_pipeop(nn_selfish)$shapes_out(list(c(NA, 4L))), "object 'self' not found")
})
