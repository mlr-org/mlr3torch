exect_torchop = function(op, inputs, task, class, exclude = character(0L), expected_class) {
  module =



  result = expect_error(op$build(inputs, task), regexp = NA,
    info = list(shapes = map(inputs, "shape"))
  )
  layer = result$layer
  expect_class(layer, class)
  output = result$output
  expect_list(output, len = op$outnum)
  expect_true(all(names(output) == op$output$name))
  expect_true(all(map_lgl(output, function(o) inherits(o, "torch_tensor"))))
  expect_set_equal(formalArgs(layer$forward), op$input$name)
  expect_true(inherits(op, "TorchOp"))

  # now we do a "param test"
  # implemented =
  init_args = setdiff(formalArgs(op$initialize), c("id", "param_vals"))
  params = op$param_set$ids()

  fn = try(getFromNamespace(class, ns = "torch"))
  if (inherits(fn, "try-error")) {
    fn = try(getFromNamespace(class, ns = "mlr3torch"))
  }
  observed = setdiff(c(init_args, params), exclude)
  expected = setdiff(formalArgs(fn), exclude)
  expect_set_equal(observed, expected)
}


autotest_torchop = function(graph, id, task, module_class = NULL) {
  result = graph$train(task)
  md = result[[1]]

  modulegraph = md$graph
  testobj = modulegraph$pipeops$merge_sum
  testmodule = testobj$module

  # class of generated module is as expected
  if (!is.null(module_class)) {
    expect_class(testmodule, c(module_class, "nn_module"))
  } else  {
    expect_class(testmodule, "nn_module")
  }

  # argument names of forward function match names of input channels
  fargs = formalArgs(testmodule$forward)
  innames = testobj$input$name
  expect_true(all(sort(fargs) == sort(innames)))

  op = pmap(graph$edges[dst_id == id, c("src_id", "src_channel")], function(src_id, src_channel) c(src_id, src_channel))
  op[[length(op) + 1L]] = md$.pointer

  # To be able to test [PipeOpTorch]s with more than one output channel we go for the list_output = TRUE
  net = model_descriptor_to_module(md, output_pointers = op, list_output = TRUE)

  ds = task_dataset(task, md$ingress)

  batch = ds$.getbatch(1)

  # this here contains the inputs that went into the module and the output.
  # It let's us verify that our shape function is correct
  out = invoke(net$forward, .args = batch$x)

  layerout = out[[md$.pointer]]
  layerin = out[[head(ob, length(op) - 1L)]]

  # we need



}
