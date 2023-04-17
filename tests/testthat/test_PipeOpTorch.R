test_that("PipeOpTorch works", {
  task = tsk("german_credit")

  # basic checks that output is checked correctly
  obj = PipeOpTorchDebug$new(id = "debug", inname = paste0("input", 1:2), outname = paste0("output", 1:2))
  expect_pipeop(obj)

  expect_error(
    PipeOpTorchDebug$new(id = "debug", inname = paste0("input", 1:2), outname = paste0("output", 1:2)),
    regexp = "grepl"
  )
  expect_error(
    PipeOpTorchDebug$new(id = "nn_debug", inname = "input", outname = paste0("output", 1:2)),
    regexp = "The names of the input channels must be a permutation of the arguments of the provided module_generator."
  )
  expect_error(
    PipeOpTorchDebug$new(id = "nn_debug", inname = "input", outname = paste0("output", 1:2)),
    regexp = "The names of the input channels must be a permutation of the arguments of the provided module_generator."
  )

  expect_error(
    PipeOpTorchDebug$new(id = "nn_debug", inname = "input", outname = paste0("output", 1:2)),
    regexp = "The names of the input channels must be a permutation of the arguments of the provided module_generator."
  )


  # basic checks that the channels are set correctly
  obj = PipeOpTorchDebug$new(id = "nn_debug", inname = paste0("input", 1:2), outname = paste0("output", 1:2))
  obj$param_set$values = list(d_out1 = 2, d_out2 = 3, bias = TRUE)

  mdin1 = po("torch_ingress_num")$train(list(task))[[1L]]
  mdin2 = po("torch_ingress_categ")$train(list(task))[[1L]]

  mdouts = obj$train(list(input1 = mdin1, input2 = mdin2))
  mdout1 = mdouts[["output1"]]

  expect_true(identical(mdout1$graph, mdout2$graph))
  expect_true(identical(mdout1$task, mdout2$task))

  expect_true(mdout1$id == "")
  mdout2 = mdouts[["output2"]]


  expect_true(list)

  expect_error(graph)

})

test_that("PipeOpTorch errs when there are unexpected NAs in the shape", {
  graph = as_graph(po("torch_ingress_num"))

  task = tsk("iris")
  md = graph$train(task)[[1L]]

  md$.pointer_shape = c(4, NA)
  expect_error(po("nn_relu")$train(list(md)), regexp = "but must have exactly one NA")

  md$.pointer_shape = c(NA, NA, 4)
  expect_error(po("nn_relu")$train(list(md)), regexp = "but must have exactly one NA")

})
