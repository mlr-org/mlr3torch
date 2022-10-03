test_that("TorchOpMerge works", {
  task = tsk("iris")
  to_add = top("add")
  to_mul = top("mul")
  expect_warning(top("cat", dim = 1L))
  expect_warning(top("cat", dim = 2L), regexp = NA)
  to_cat1 = suppressWarnings(top("cat", dim = 1L))
  to_cat2 = suppressWarnings(top("cat", dim = 1L))
  x1 = torch_randn(1, 3)
  x2 = torch_randn(1, 3)
  y = torch_randn(1)
  inputs = list(input1 = x1, input2 = x2)

  c(layer, output) %<-% top("add")$build(inputs, task)
  expect_true(torch_equal(x1 + x2, output$output))

  c(layer, output) %<-% top("mul")$build(inputs, task)
  expect_true(torch_equal(x1 * x2, output$output))

  c(layer, output) %<-% suppressWarnings(top("cat", dim = 2L))$build(inputs, task)
  expect_true(torch_equal(torch_cat(list(x1, x2), dim = 2L), output$output))


  c(layer, output) %<-% suppressWarnings(top("cat", dim = 1L))$build(inputs, task)
  expect_true(torch_equal(torch_cat(list(x1, x2), dim = 1L), output$output))
})

# TODO: Also test for order here

test_that("TorchOpMergeAdd works", {
  task = tsk("iris")
  graph = gunion(list(pot("ingress_num_1"), pot("ingress_num_2"))) %>>%
    pot("merge_sum", innum = 2L)

  graph = graph
  id = "merge_sum"
  module_class = "nn_merge_sum"
  task = tsk("iris")

  autotest_torchop(
    graph = graph,
    id = "merge_sum",
    module_class = "nn_merge_sum",
    task = tsk("iris")
  )

  out = graph$train(task)
  g = md$graph

  testobj = g$pipeops$merge_sum

  fargs = formalArgs(testobj$module$forward)
  innames = testobj$input$name

  expect_true(all(sort(fargs) == sort(innames)))
})

