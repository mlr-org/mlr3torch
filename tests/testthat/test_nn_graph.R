test_that("nn_graph basic checks", {
  task = tsk("iris")
  md = po("torch_ingress_num")$train(list(task))[[1L]]
  shapes_in = map(md$ingress, "shape")
  graph = md$graph
  input = sample_input_from_shapes(shapes_in)

  # a very basic check with list_output = FALSE (the default)
  network = nn_graph(graph, shapes_in)
  expect_class(network, "nn_graph")
  output = invoke(network, .args = input)
  expect_identical(input[[1L]], output)
  expect_identical(network$graph, graph)
  expect_identical(network$graph_input_name, graph$input$name)
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

  expect_deep_clone(network$graph, network1$graph)
  network$graph = NULL
  network1$graph = NULL
  expect_deep_clone(network, network1)
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

  expect_deep_clone(nn_test, nn_test1)
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
