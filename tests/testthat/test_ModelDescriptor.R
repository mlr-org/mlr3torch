test_that("ModelDescriptor basic checks", {
  out_features = 3
  g = po("torch_ingress_num") %>>%
    po("nn_linear", out_features = out_features)
  task = tsk("iris")
  optimizer = t_opt("adam")
  loss = t_loss("cross_entropy")
  cb = t_clbk("checkpoint")

  ingress = TorchIngressToken(
    features = task$feature_names,
    batchgetter = batchgetter_num,
    shape = c(NA, 4)
  )

  ingresslist = list(torch_ingress_num.input = ingress)
  pointer = c("nn_linear", "output")
  pointer_shape = c(NA, out_features)
  
  md = ModelDescriptor(
    graph = g,
    ingress = ingresslist,
    task = task,
    loss = loss,
    optimizer = optimizer,
    callbacks = list(cb),
    .pointer = pointer,
    .pointer_shape = pointer_shape
  )

  expect_class(md, "ModelDescriptor")
  expect_identical(md$graph, g)
  expect_identical(md$task, task)
  expect_identical(md$ingress, ingresslist)
  expect_identical(md$callbacks, list(cb))
  expect_identical(md$.pointer, pointer)
  expect_identical(md$.pointer_shape, pointer_shape)

  repr = capture.output(md)

  a = R6Class("A")$new()

  expect_error(ModelDescriptor(
      graph = g,
      ingress = list(a = ingress),
      task = task,
      loss = t_loss("mse")
    ),
    regexp = "Names must be a subset of {'torch_ingress_num.input'}, but has additional elements {'a'}.",
    fixed = TRUE
  )

  expect_error(ModelDescriptor(
      graph = g,
      ingress = list(a = ingress),
      task = task
    ),
    regexp = "Names must be a subset of {'torch_ingress_num.input'}, but has additional elements {'a'}.",
    fixed = TRUE
  )

  expect_error(ModelDescriptor(
      graph = g,
      ingress = list(torch_ingress_num.input = ingress),
      task = a
    ),
    regexp = "Must inherit from class 'Task'",
    fixed = TRUE
  )
  expect_error(ModelDescriptor(
      graph = a,
      ingress = list(torch_ingress_num.input = ingress),
      task = task
    ),
    regexp = "Must inherit from class 'Graph'",
    fixed = TRUE
  )

  expect_error(ModelDescriptor(
      graph = g,
      ingress = list(torch_ingress_num.input = ingress),
      task = task,
      loss = a
    ),
    regexp = "Must inherit from class 'TorchLoss'",
    fixed = TRUE
  )

  expect_error(ModelDescriptor(
      graph = g,
      ingress = list(torch_ingress_num.input = ingress),
      task = task,
      optimizer = a
    ),
    regexp = "Must inherit from class 'TorchOptimizer'",
    fixed = TRUE
  )

  expect_error(ModelDescriptor(
      graph = g,
      ingress = list(torch_ingress_num.input = ingress),
      task = task,
      callbacks = list(cb, cb)
    ),
    regexp = "Must have unique names",
    fixed = TRUE
  )
})

# test_that("union works", {
#   task = tsk("iris")
#   g1 = po("torch_ingress_num") %>>%
#     po("nn_linear", out_features = 10)
#
#   g2 = po("torch_ingress_categ") %>>%
#     po("nn_linear", out_features = 10)
#
#
#   md1 = ModelDescriptor(
#     graph = g1,
#
#   )
#   md 
# })
