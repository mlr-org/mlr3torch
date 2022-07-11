expect_torchop = function(op, inputs, task, param_vals) {
  op$param_set$values = insert_named(op$param_set$values, param_vals)
  result = expect_error(op$build(inputs, task), regexp = NA,
    info = list(shapes = map(inputs, "shape"))
  )
  layer = result$layer
  output = result$output

  expect_list(output, len = op$outnum)
  expect_true(all(names(output) == op$output$name))
  expect_true(all(map_lgl(output, function(o) inherits(o, "torch_tensor"))))
  expect_set_equal(formalArgs(layer$forward), op$input$name)
  expect_true(inherits(op, "TorchOp"))
}
