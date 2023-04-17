test_that("nn_graph basic checks", {
  task = tsk("iris")
  md = po("torch_ingress_num")$train(list(task))[[1L]]
  shapes_in = map(md$ingress, "shape")
  graph = md$graph
  input = sample_input_from_shapes(shapes_in)

  network = nn_graph(graph, shapes_in)
  expect_class(network, "nn_graph")
  output = invoke(network, .args = input)
  expect_identical(input[[1L]], output)

  expect_identical(network$graph, graph)
  expect_identical(network$graph_input_name, graph$input$name)
  expect_identical(network$list_output, FALSE)

  network_l = nn_graph(graph, shapes_in, list_output = TRUE)
  expect_class(network_l, "nn_graph")

  output_l = invoke(network_l, .args = input)
  expect_identical(unname(input), unname(output_l))
  expect_identical(names(output_l), graph$output$name)

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

  po_debug = PipeOpTorchDebug$new(param_vals = list(d_out1 = 1, d_out2 = 2))
  graph2 = list(
    po("torch_ingress_num_1"),
    po("torch_ingress_num_2")
    ) %>>%
    po_debug

  md2 = graph2$train(task)[[1L]]
  graph2 = md2$graph
  shapes_in2 = map(md2$ingress, "shape")
  
  network2 = nn_graph(graph2, shapes_in = shapes_in2, list_output = TRUE)
  output2 = invoke(network2, .args = input1)


  expect_equal(
    names(output2),
    c("nn_debug.output1", "nn_debug.output2")
  )
  expect_equal(output2[[1]]$shape, c(1, 1))
  expect_equal(output2[[2]]$shape, c(1, 2))

})

test_that("nn_graph verifies inputs correctly", {

})

test_that("nn_graph can reset parameters", {

})

test_that("argument_matcher works", {

})

test_that("unique_id works", {

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

  input = sample_input_from_shapes(net$shapes_in)
})

test_that("model_descriptor_to_learner works", {
  task = tsk("iris")

  graph1 = po("torch_ingress_num") %>>%
    po("nn_linear", out_features = 10) %>>%
    po("nn_relu") %>>%
    po("nn_head") %>>%
    po("torch_optimizer") %>>%
    po("torch_loss", loss = "mse") %>>%
    po("torch_callbacks")

  md = graph1$train(task)[[1L]]

  learner = model_descriptor_to_learner(md)

  expect_class(learner, "LearnerClassifTorchModel")

  learner$param_set$set_values(batch_size = 150, epochs = 0)

  expect_learner_torch(learner)

  ids = partition(task)

  learner$train(task, row_ids = ids$train)
  pred = learner$predict(task, row_ids = ids$test)
  expect_class(pred, "PredictionClassif")
})

# test_that("Linear graph", {
#   batch_size = 16L
#   d_token = 3L
#   task = tsk("iris")
#
#   graph = po("torch_ingress_num") %>>%
#     po("nn_linear", out_features = 10) %>>%
#     po("nn_relu") %>>%
#     po("nn_head")
#
#   md = graph$train(task)[[1L]]
#   network = md
#   expect_class(md, "ModelDescriptor")
#
#   network = md$graph
#   expect_class(network, "nn_graph")
#
#   network = graph$train(task)[[1L]][[2L]]
#   expect_function(network)
#   expect_true(inherits(network, "nn_graph"))
# })
#
# test_that("Forking of depth 1 produces graph", {
#   batch_size = 9L
#   task = tsk("iris")
#   graph = po("torch_ingress_num") %>>%
#     list(
#       a = po("nn_linear_1", out_features = 3L) %>>% po("nn_relu"),
#       b = po("nn_linear_2", out_features = 3L)
#     ) %>>%
#     po("nn_merge_sum")
#   md = graph$train(task)[[1L]]
#   expect_class(md$graph, "nn_graph")
# })
#
# test_that("Forking of depth 2 produces graph", {
#   #
#   #                                  --> aa.linear -->
#   #                      --> a.linear
#   # tokenizer --> flatten            --> ab.linear --> merge
#   #
#   #                      --> b.linear --------------->
#   d_token = 4L
#   batch_size = 9L
#   task = tsk("iris")
#   batch = get_batch(task, batch_size, device = "cpu")
#   a = gunion(
#     graphs = list(
#       c = top("linear", out_features = 3L),
#       d = top("linear", out_features = 3L)
#     )
#   ) %>>%
#     top("merge", .method = "mul", innum = 2L)
#
#
#   graph = top("input") %>>%
#     top("select", items = "num") %>>%
#     gunion(
#       graphs = list(
#         a = a,
#         b = top("linear", out_features = 3L)
#       )
#     ) %>>%
#     top("merge", .method = "add") %>>%
#     top("linear", out_features = 1L)
#   network = graph$train(task)[[1L]][[2L]]
#   expect_function(network)
#   expect_true(inherits(network, "nn_graph"))
# })
#
# test_that("fork at the beginning works", {
#   graph = top("input") %>>%
#     top("select", items = "num") %>>%
#     gunion(list(top("linear_1", out_features = 10), top("linear_2", out_features = 10))) %>>%
#     top("add")
#
#   task = tsk("iris")
#   res = graph$train(task)[[1L]]
#   net = res$network
#   net(list(num = torch_randn(16, 4)))
# })
#
#
# FIXME: Need to wait for the pipelines PR
# test_that("would work with multi-output torchops", {
#   TorchOpMO = R6Class("TorchOpMO",
#     inherit = TorchOp,
#     public = list(
#       initialize = function(id = "mo", param_vals = list()) {
#         output = data.table(
#           name = c("output1", "output2"),
#           train = c("ModelConfig", "ModelConfig"),
#           predict = c("Task", "Task")
#         )
#         super$initialize(
#           param_set = ps(),
#           param_vals = param_vals,
#           id = id,
#           output = output
#         )
#       }
#     ),
#     private = list(
#       .build = function(inputs, task) {
#         nn_module("multioutput",
#           forward = function(input) {
#             list(output2 = input - 100, output1 = input + 100)
#           }
#         )()
#       }
#     )
#   )
#
#   op = TorchOpMO$new()
#
#   graph = top("input") %>>%
#     top("select", items = "num") %>>%
#     op
#
#   graph$add_pipeop(top("cat", innum = 2L, dim = 2L))
#   graph2 = graph$clone(deep = TRUE)
#
#   graph$add_edge("mo", "cat", "output1", "input1")
#   graph$add_edge("mo", "cat", "output2", "input2")
#   graph$add_edge("mo", "cat", "output2", "input3")
#
#
#   graph2$add_edge("mo", "cat", "output1", "input2")
#   graph2$add_edge("mo", "cat", "output2", "input1")
#
#
#
#   task = tsk("iris")
#
#   net1 = graph$train(task)[[1L]]$network
#   net2 = graph2$train(task)[[1L]]$network
#
#   x = torch_randn(1, 4)
#   observed1 = net1(list(num = x))
#   observed2 = net2(list(num = x))
#   expected = 2 * x
#   expect_true(torch_equal(observed, expected))
#
#
# })



