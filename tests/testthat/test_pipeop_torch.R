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
  # `initialize` and `forward` become methods of the module and are therefore evaluated in
  # `parent_env`, which defaults to the caller -- the frame of `make()` here, where they are also
  # written. `shapes_out()` is not a method and keeps its own environment in either case.
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

  # ... `shapes_out()` also when it is written somewhere other than the call site
  shapes_out = local({
    k = 2L
    function(shapes_in) list(c(shapes_in[[1L]][1L], k))
  })
  obj = pipeop_torch("nn_x", forward = function(input) input, shapes_out = shapes_out)$new()
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA, 2L)))

  # `forward` does not, so a wrapper around `pipeop_torch()` -- whose frame is the default
  # `parent_env`, but not where the functions are written -- has to pass `parent_env` along
  wrapper = function(...) pipeop_torch("nn_y", ...)
  make_wrapped = function(k, pass_env = FALSE) {
    forward = function(input) input[, 1:k]
    shapes_out = function(shapes_in) list(c(shapes_in[[1L]][1L], k))
    if (pass_env) {
      wrapper(forward = forward, shapes_out = shapes_out, parent_env = environment())
    } else {
      wrapper(forward = forward, shapes_out = shapes_out)
    }
  }
  build = function(obj) {
    model_descriptor_to_module((po("torch_ingress_num") %>>% obj)$train(tsk("iris"))[[1L]])
  }
  # the closure is only missed when the module runs, which is why it is worth a test
  network = build(make_wrapped(2L)$new())
  expect_error(with_no_grad(network(torch_ingress_num.input = torch_randn(3L, 4L))),
    "object 'k' not found")

  network = build(make_wrapped(2L, pass_env = TRUE)$new())
  expect_equal(dim(with_no_grad(network(torch_ingress_num.input = torch_randn(3L, 4L)))), c(3L, 2L))
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

test_that("shapes_out() has to return shapes", {
  # the shapes are coerced with `as.integer()`, which would turn these into `NA`s instead of failing
  for (bad in list(quote(list("a")), quote(list(NULL)), quote(list(list(1, 2))))) {
    obj = pipeop_torch("nn_c", forward = function(input) input,
      shapes_out = eval(bquote(function(shapes_in) .(bad))))$new()
    expect_error(obj$shapes_out(list(c(NA, 4L))), "invalid shape for output channel 'output'",
      fixed = TRUE)
  }

  # a shape whose dimensions are all unknown is valid
  obj = pipeop_torch("nn_d", forward = function(input) input,
    shapes_out = function(shapes_in) list(c(NA, NA)))$new()
  expect_equal(obj$shapes_out(list(c(NA, 4L))), list(output = c(NA_integer_, NA_integer_)))
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
  # a `ParamSet` that does not cover a required argument would only fail during training
  expect_error(pipeop_torch("nn_e",
    initialize = function(shapes_in, a, b) NULL,
    forward = function(input) input, shapes_out = function(shapes_in) shapes_in,
    param_set = ps(a = p_uty(tags = "train"))), "no parameter(s) 'b'", fixed = TRUE)
  # ... one with a default can be left to the module
  expect_class(pipeop_torch("nn_f",
    initialize = function(shapes_in, a, b = 1) NULL,
    forward = function(input) input, shapes_out = function(shapes_in) shapes_in,
    param_set = ps(a = p_uty(tags = "train"))), "R6ClassGenerator")

  # `shapes_out()` gets only what it declares, so an argument it cannot get is a typo
  expect_error(pipeop_torch("nn_g", forward = function(input) input,
    shapes_out = function(shape_in) shape_in), "arguments of `shapes_out()`", fixed = TRUE)

  # the id has to be one a class name can be derived from, unless it is given
  expect_error(pipeop_torch("", forward = function(input) input,
    shapes_out = function(shapes_in) shapes_in), "at least 1 characters")
  expect_error(pipeop_torch("nn_", forward = function(input) input,
    shapes_out = function(shapes_in) shapes_in), "No class name can be derived")
  expect_equal(pipeop_torch("nn_", forward = function(input) input,
    shapes_out = function(shapes_in) shapes_in, classname = "PipeOpTorchFoo")$classname,
    "PipeOpTorchFoo")
  expect_equal(pipeop_torch("nn_multihead_attention", forward = function(input) input,
    shapes_out = function(shapes_in) shapes_in)$classname, "PipeOpTorchMultiheadAttention")
})

test_that("the input channels have to match forward()", {
  # the module is called with one argument per input channel, so a mismatch would only surface when
  # the network is run
  expect_error(pipeop_torch("nn_h", forward = function(input) input, in_channels = 2L,
    shapes_out = function(shapes_in) shapes_in[1L]), "takes 1 argument(s) (input)", fixed = TRUE)
  expect_error(pipeop_torch("nn_i", forward = function(a, b) a, in_channels = 1L,
    shapes_out = function(shapes_in) shapes_in), "2 argument(s) (a, b)", fixed = TRUE)
  # an argument with a default may be left to the module, and a vararg forward() takes anything
  expect_class(pipeop_torch("nn_j", forward = function(a, b = 1) a, in_channels = 1L,
    shapes_out = function(shapes_in) shapes_in), "R6ClassGenerator")
  expect_class(pipeop_torch("nn_k", forward = function(...) ..1, in_channels = 3L,
    shapes_out = function(shapes_in) shapes_in[1L]), "R6ClassGenerator")

  # a vararg channel is only meaningful on the input side, and only on its own
  expect_error(pipeop_torch("nn_l", forward = function(input) input, out_channels = "...",
    shapes_out = function(shapes_in) shapes_in), "not a valid output channel name")
  expect_error(pipeop_torch("nn_n", forward = function(...) ..1, in_channels = c("...", "a"),
    shapes_out = function(shapes_in) shapes_in), "must be the only input channel")
})
