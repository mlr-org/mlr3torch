test_that("nn_graph basic checks", {
  task = tsk("iris")
  md = po("torch_ingress_num")$train(list(task))[[1L]]
  shapes_in = map(md$ingress, "shape")
  graph = md$graph
  input = sample_input_from_shapes(shapes_in)

  # a very basic check with list_output = FALSE (the default)
  network = nn_graph(graph, shapes_in)
  expect_equal(network$input_map, "torch_ingress_num.input")
  expect_class(network, "nn_graph")
  output = invoke(network, .args = input)
  expect_identical(input[[1L]], output)
  expect_identical(network$graph, graph)
  expect_identical(network$input_map, graph$input$name)
  expect_identical(network$list_output, FALSE)

  # a very basic check with list_output = TRUE
  network_l = nn_graph(graph, shapes_in, list_output = TRUE)
  expect_class(network_l, "nn_graph")
  output_l = invoke(network_l, .args = input)
  expect_identical(unname(input), unname(output_l))
  expect_identical(names(output_l), graph$output$name)

  # a basic check with two inputs but no actual computation (output is equal to input)
  mds = as_graph(list(po("torch_ingress_num_1"), po("torch_ingress_num_2")))$train(task)
  md1 = model_descriptor_union(mds[[1]], mds[[2]])

  shapes_in1 = map(md1$ingress, "shape")
  graph1 = md1$graph

  network1 = nn_graph(graph1, shapes_in1, list_output = TRUE)

  input1 = sample_input_from_shapes(shapes_in1)
  output1 = invoke(network1, .args = input1)
  expect_identical(
    names(output1),
    c("torch_ingress_num_1.output", "torch_ingress_num_2.output")
  )
  expect_identical(input1[[1]], output1[[1]])
  expect_identical(input1[[2]], output1[[2]])

  # Two inputs and two outputs and some actual computation
  po_debug = PipeOpTorchDebug$new(param_vals = list(d_out1 = 1, d_out2 = 2))
  # input and shapes stay the same
  graph2 = list(
    po("torch_ingress_num_1"),
    po("torch_ingress_num_2")
    ) %>>%
    po_debug

  md2 = graph2$train(task)[[1L]]
  graph2 = md2$graph
  shapes_in2 = map(md2$ingress, "shape")
  network2 = nn_graph(graph2, shapes_in = shapes_in1, list_output = TRUE)
  output2 = invoke(network2, .args = input1)

  expect_equal(
    names(output2),
    c("nn_debug.output1", "nn_debug.output2")
  )
  expect_equal(output2[[1]]$shape, c(1, 1))
  expect_equal(output2[[2]]$shape, c(1, 2))
})

test_that("nn_graph can reset parameters", {
  nn_mul = nn_module("nn_mul",
    initialize = function(net) {
      self$a = nn_parameter(torch_tensor(-99))
    },
    forward = function(x) {
      x * self$a
    },
    reset_parameters = function() {
      self$a = nn_parameter(torch_tensor(1))
    }
  )

  nn_add = nn_module("nn_add",
    initialize = function() {
      self$a = nn_parameter(torch_tensor(-99))
    },
    forward = function(x) {
      x + a
    },
    reset_parameters = function() {
      self$a = nn_parameter(torch_tensor(0))
    }
  )

  nn_lin = nn_module("nn_lin",
    initialize = function() {
      self$nn_add = nn_add()
      self$nn_mul = nn_mul()
    },
    forward = function(x) {
      self$nn_add(self$nn_mul(x))
    }
  )

  network = nn_graph(as_graph(po("module", module = nn_lin()), clone = FALSE),
    shapes_in = list(module.input = 1)
  )

  expect_equal(network$parameters[["module_list.0.nn_add.a"]]$item(), -99)
  expect_equal(network$parameters[["module_list.0.nn_mul.a"]]$item(), -99)

  network$reset_parameters()

  expect_equal(network$parameters[["module_list.0.nn_add.a"]]$item(), 0)
  expect_equal(network$parameters[["module_list.0.nn_mul.a"]]$item(), 1)
})

test_that("argument_matcher works", {
  am = argument_matcher(letters[1:3])
  res = am(c = 3, a = 1, b = 2)

  expect_equal(res, list(a = 1, b = 2, c = 3))
})

test_that("unique_id works", {
  expect_equal(unique_id("a", c("a", "b")), "a_1")
  expect_equal(unique_id("a", c("a", "b", "a_1")), "a_2")
})

test_that("model_descriptor_to_module works", {
  task = tsk("iris")

  graph1 = po("torch_ingress_num") %>>%
    po("nn_linear", out_features = 10) %>>%
    po("nn_relu") %>>%
    po("nn_head")

  md = graph1$train(task)[[1L]]

  net = model_descriptor_to_module(md, list(c("nn_head", "output")))
  batch = sample_input_from_shapes(net$shapes_in)
  invoke(net, .args = batch)
  result1 = net(torch_ingress_num.input = batch[[1L]])

  expect_equal(result1$shape, c(1, 3))

  in_sepal = po("select_1", selector = selector_grep("Sepal")) %>>% po("torch_ingress_num_1")
  in_petal = po("select_2", selector = selector_grep("Petal")) %>>% po("torch_ingress_num_2")

  graph2 = list(in_sepal, in_petal) %>>%
    po("nn_merge_sum") %>>%
    po("nn_head")

  md = graph2$train(task)[[1L]]

  net = model_descriptor_to_module(md, list(c("nn_head", "output")))
  expect_class(net, "nn_graph")

  input = sample_input_from_shapes(net$shapes_in)

  batch = sample_input_from_shapes(net$shapes_in)
  result2 = net(torch_ingress_num_1.input = batch[[1L]], torch_ingress_num_2.input = batch[[2L]])

  expect_equal(result2$shape, c(1, 3))
})

test_that("cloning", {
  task = tsk("iris")

  graph1 = po("torch_ingress_num") %>>%
    po("nn_linear", out_features = 10) %>>%
    po("nn_relu") %>>%
    po("nn_head") %>>%
    po("torch_optimizer") %>>%
    po("torch_loss", loss = "cross_entropy") %>>%
    po("torch_callbacks")

  md = graph1$train(task)[[1L]]

  learner = model_descriptor_to_learner(md)

  expect_class(learner, "LearnerTorchModel")

  learner$param_set$set_values(batch_size = 150, epochs = 0)

  expect_learner_torch(learner, task = tsk("iris"))

  ids = partition(task)

  learner$train(task, row_ids = ids$train)
  network = learner$network

  network1 = learner$network$clone(deep = TRUE)

  for (net in list(network, network1)) {
    for (pipeop in net$graph$pipeops) {
      unlockBinding(".additional_phash_input", get_private(pipeop))
      get_private(pipeop, ".additional_phash_input") = function(...) NULL
    }
  }


  expect_false(identical(network$module_list, network1$module_list))
  expect_false(identical(network$graph, network1$graph))
  expect_false(identical(network$graph$pipeops, network1$graph$pipeops))
  # first module is self
  expect_false(identical(network$module_list$modules[[2]], network1$module_list$modules[[2]]))
  expect_false(identical(network$graph$pipeops$nn_linear$module, network1$graph$pipeops$nn_linear$module))
  # references are preserved between the graph and the module list
  # the first list entry of module list is the module list itself
  expect_true(identical(network1$graph$pipeops$nn_linear$module, network1$module_list$modules[[2]]))

  expect_deep_clone_mlr3torch(network$graph, network1$graph)
  network$graph = NULL
  network1$graph = NULL
  expect_deep_clone_mlr3torch(network, network1)
})

test_that("cloning", {
  nn_test = nn_module("test", initialize = function() {
    self$l = nn_module_list(list(nn_linear(1, 1)))
    },
    forward = function(x) {
      self$l[[1]](x)
    }
  )()

  nn_test1 = nn_test$clone(deep = TRUE)
  nn_test = nn_test$clone(deep = TRUE)$clone(deep = TRUE)

  expect_deep_clone_mlr3torch(nn_test, nn_test1)
  nn_test = nn_module("test", initialize = function() {
    self$l = nn_module_list(list(nn_linear(1, 1)))
    },
    forward = function(x) {
      self$l[[1]](x)
    }
  )()

  nn_test1 = nn_test$clone(deep = TRUE)
  nn_test$clone(deep = TRUE)

  nn_test$l$modules[[2]]
  nn_test1$l$modules[[2]]

  identical(nn_test$l$modules[[2]], nn_test1$l$modules[[2]])
  identical(nn_test$children$l, nn_test1$children$l)
})

test_that("non-terminal output", {
  md = (po("torch_ingress_num") %>>% po("nn_head") %>>% po("nn_reshape", shape = c(-1, 1, 3)))$train(tsk("iris"))[[1L]]
  module = model_descriptor_to_module(md, list(c("nn_head", "output")), list_output = TRUE)
  x = torch_randn(1, 4)
  xout = module(x)
  expect_equal(xout[[1L]]$shape, c(1, 3))
  expect_equal(names(xout), "output_nn_head.output")
  expect_true("output_nn_head.output" %in% module$graph$output$name)
})

forward_plan = function(network) network$.__enclos_env__$private$.forward_plan

test_that("forward plan gives the same result as Graph$train()", {
  task = tsk("iris")
  compare = function(network, input) {
    expected = network$graph$train(unname(input), single_input = FALSE)[network$output_map]
    if (!network$list_output) expected = expected[[1L]]
    expect_equal(invoke(network, .args = input), expected)
    expect_list(forward_plan(network))
  }

  # a chain with a block, as created by nn() and model_descriptor_to_module()
  md = (po("torch_ingress_num") %>>% nn("linear", out_features = 8) %>>%
    nn("block", nn("linear", out_features = 8) %>>% nn("relu"), n_blocks = 2) %>>% nn("head"))$train(task)[[1L]]
  network = model_descriptor_to_module(md)
  compare(network, sample_input_from_shapes(network$shapes_in))

  # two inputs and a module with two outputs
  po_debug = PipeOpTorchDebug$new(param_vals = list(d_out1 = 1, d_out2 = 2))
  md2 = (list(po("torch_ingress_num_1"), po("torch_ingress_num_2")) %>>% po_debug)$train(task)[[1L]]
  network2 = nn_graph(md2$graph, shapes_in = map(md2$ingress, "shape"), list_output = TRUE)
  compare(network2, sample_input_from_shapes(network2$shapes_in))

  # the output of one module is used by two other modules, which are merged via a vararg channel,
  # and also returned
  md3 = (po("torch_ingress_num") %>>% nn("linear", out_features = 3) %>>%
    gunion(list(nn("relu"), nn("sigmoid"))) %>>% nn("merge_sum"))$train(task)[[1L]]
  network3 = model_descriptor_to_module(md3, list(md3$pointer, c("linear", "output")), list_output = TRUE)
  compare(network3, sample_input_from_shapes(network3$shapes_in))
  expect_equal(names(network3(torch_randn(2, 4))), network3$output_map)

  # the output map can be changed after the plan was created
  network3$output_map = rev(network3$output_map)
  compare(network3, sample_input_from_shapes(network3$shapes_in))
})

test_that("nn_graph falls back to Graph$train() for operators other than PipeOpModule and PipeOpNOP", {
  graph = po("module", id = "linear", module = nn_linear(2, 3)) %>>% po("copy", outnum = 2)
  network = nn_graph(graph, shapes_in = set_names(list(c(NA, 2)), graph$input$name), list_output = TRUE)
  out = network(torch_randn(4, 2))
  expect_false(forward_plan(network))
  expect_equal(names(out), c("copy.output1", "copy.output2"))
  expect_equal(out[[1L]], out[[2L]])
})

test_that("assigning a graph resets the forward plan", {
  graph = as_graph(po("module", id = "linear", module = nn_linear(2, 3)))
  network = nn_graph(graph, shapes_in = set_names(list(c(NA, 2)), graph$input$name))
  x = torch_randn(4, 2)
  network(x)
  expect_list(forward_plan(network))
  network$graph = po("module", id = "linear", module = nn_linear(2, 3)) %>>% po("module", id = "relu", module = nn_relu())
  expect_null(forward_plan(network))
  network$output_map = "relu.output"
  expect_true(all(as_array(network(x)) >= 0))
  expect_equal(forward_plan(network)$steps[[2L]]$id, "relu")
})

test_that("forward plan uses the modules of a deep clone", {
  md = (po("torch_ingress_num") %>>% nn("linear", out_features = 3))$train(tsk("iris"))[[1L]]
  network = model_descriptor_to_module(md)
  x = torch_randn(2, 4)
  network(x)
  network1 = network$clone(deep = TRUE)
  expect_equal(network(x), network1(x))
  with_no_grad(network1$graph$pipeops$linear$module$weight$add_(1))
  expect_false(torch_equal(network(x), network1(x)))
  expect_true(torch_equal(network1(x), network1$graph$train(list(x), single_input = FALSE)[[1L]]))
})

test_that("the forward plan is created anew when PipeOps are added to the graph in place", {
  graph = po("module", id = "linear", module = nn_linear(2, 3)) %>>% po("module", id = "relu", module = nn_relu())
  network = nn_graph(graph, shapes_in = set_names(list(c(NA, 2)), graph$input$name), output_map = "relu.output")
  x = torch_randn(4, 2)
  network(x)
  network$graph$add_pipeop(po("module", id = "double", module = function(x) x * 2))
  network$graph$add_edge("relu", "double")
  network$output_map = "double.output"
  expect_equal(network(x), network$graph$train(list(x), single_input = FALSE)[["double.output"]])
  expect_equal(forward_plan(network)$pipeop_ids, names(network$graph$pipeops))
})
