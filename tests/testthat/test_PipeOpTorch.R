test_that("Basic checks", {
  task = tsk("german_credit")

  # basic checks that output is checked correctly
  obj = PipeOpTorchDebug$new(id = "debug", inname = paste0("input", 1:2), outname = paste0("output", 1:2))
  expect_pipeop(obj)
  expect_class(obj, "PipeOpTorch")

  expect_equal(unique(obj$input$train), "ModelDescriptor")
  expect_equal(unique(obj$output$train), "ModelDescriptor")

  expect_equal(unique(obj$input$predict), "Task")
  expect_equal(unique(obj$output$predict), "Task")
  expect_class(obj$module_generator, "nn_module_generator")
  expect_equal(obj$tags, "torch")
  expect_set_equal(obj$packages, c("mlr3torch", "torch", "mlr3pipelines"))
})

test_that("cloning works", {
  # can't use PipeOpTorchDebug because it then fails for some reason because of a  missing help page
  obj = PipeOpTorchReLU$new()
  obj1 = obj$clone(deep = TRUE)
  expect_deep_clone(obj, obj1)
})

test_that("single input and output", {
  task = tsk("iris")

  # train
  md = (po("torch_ingress_num") %>>%
    po("torch_optimizer") %>>%
    po("torch_loss", "cross_entropy") %>>%
    po("torch_callbacks", "checkpoint"))$train(task)[[1L]]

  obj = po("nn_linear", out_features = 10)

  mdout = obj$train(list(md))[[1L]]
  expect_identical(address(md$graph), address(mdout$graph))
  expect_true(!identical(md$pointer, mdout$pointer))
  expect_true(!identical(md$pointer_shape, mdout$pointer_shape))
  expect_equal(address(md$loss), address(mdout$loss))
  expect_equal(address(md$optimizer), address(mdout$optimizer))
  expect_equal(address(md$callbacks[[1L]]), address(mdout$callbacks[[1L]]))
  expect_equal(mdout$pointer, c("nn_linear", "output"))
  expect_equal(mdout$pointer_shape, c(NA, 10))
  expect_true(obj$is_trained)
  expect_true("nn_linear" %in% names(mdout$graph$pipeops))
  expect_class(mdout$graph$pipeops$nn_linear, "PipeOpModule")
  expect_class(mdout$graph$pipeops$nn_linear$module, "nn_linear")
  expect_equal(
    data.table(
      src_id = "torch_ingress_num",
      src_channel = "output",
      dst_id = "nn_linear",
      dst_channel = "input"
    ),
    mdout$graph$edges
  )

  # predict
  taskout = obj$predict(list(task))
  expect_identical(address(taskout[[1L]]), address(taskout[[1L]]))
})

test_that("train handles multiple input channels correctly", {
  task = tsk("iris")

  # first we start with vararg
  obj = po("nn_merge_sum")

  graph = as_graph(list(
    po("select_1", selector = selector_grep("Sepal")) %>>% po("torch_ingress_num_1"),
    po("select_2", selector = selector_grep("Petal")) %>>% po("torch_ingress_num_2"))
  )

  mds = graph$train(task)
  mdsout = obj$train(mds)
  expect_true(obj$is_trained)
  expect_equal(address(mdsout[[1L]]$graph), address(mdsout[[1L]]$graph))
  expect_equal(mdsout[[1L]]$pointer, c("nn_merge_sum", "output"))
  expect_equal(mdsout[[1L]]$pointer_shape, c(NA, 2))

  expect_equal(
    data.table(
      src_id = c("torch_ingress_num_1", "torch_ingress_num_2"),
      src_channel = c("output", "output"),
      dst_id = "nn_merge_sum",
      dst_channel = c("...", "...")
    ),
    mdsout[[1L]]$graph$edges
  )


  # two inputs two outputs


  obj = PipeOpTorchDebug$new(id = "nn_debug", inname = paste0("input", 1:2), outname = paste0("output", 1:2))
  obj$param_set$set_values(d_out1 = 2, d_out2 = 3, bias = TRUE)

  mdin1 = (po("select", selector = selector_grep("Petal")) %>>% po("torch_ingress_num_1"))$train(task)[[1L]]
  mdin2 = (po("select", selector = selector_grep("Sepal")) %>>% po("torch_ingress_num_2"))$train(task)[[1L]]

  mdouts = obj$train(list(input1 = mdin1, input2 = mdin2))
  mdout1 = mdouts[["output1"]]
  mdout2 = mdouts[["output2"]]

  expect_equal(address(mdout1$graph), address(mdout2$graph))
  expect_equal(mdout1$pointer, c("nn_debug", "output1"))
  expect_equal(mdout2$pointer, c("nn_debug", "output2"))
  expect_equal(mdout1$pointer_shape, c(NA, 2))
  expect_equal(mdout2$pointer_shape, c(NA, 3))
})

test_that("shapes_out", {
  obj = po("nn_linear", out_features = 3)

  # single input
  expect_equal(obj$shapes_out(c(NA, 1)), list(output = c(NA, 3)))
  expect_equal(obj$shapes_out(list(c(NA, 1))), list(output = c(NA, 3)))
  expect_equal(obj$shapes_out(list(input = c(NA, 1))), list(output = c(NA, 3)))
  expect_equal(obj$shapes_out(list(x = c(NA, 1))), list(output = c(NA, 3)))

  # multiple inputs
  obj1 = PipeOpTorchDebug$new()
  obj1$param_set$set_values(d_out1 = 2, d_out2 = 3)

  expect_equal(obj1$shapes_out(list(c(NA, 99), c(NA, 3))), list(output1 = c(NA, 2), output2 = c(NA, 3)))
  expect_error(obj1$shapes_out(list(c(NA, 99))), regexp = "number of input")
})

test_that("PipeOpTorch errs when there are unexpected NAs in the shape", {
  graph = as_graph(po("torch_ingress_num"))

  task = tsk("iris")
  md = graph$train(task)[[1L]]

  md$pointer_shape = c(4, NA)
  expect_error(po("nn_relu")$train(list(md)), regexp = "Invalid shape")

  md$pointer_shape = c(NA, NA, 4)
  expect_error(po("nn_relu")$train(list(md)), regexp = "Invalid shape")

})
