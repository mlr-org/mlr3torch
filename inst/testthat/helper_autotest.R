expect_torchop = function(op, inputs, task, class, exclude = character(0L)) {
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

