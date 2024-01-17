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
    pointer = pointer,
    pointer_shape = pointer_shape
  )

  expect_class(md, "ModelDescriptor")
  expect_identical(md$graph, g)
  expect_identical(md$task, task)
  expect_identical(md$ingress, ingresslist)
  expect_identical(md$callbacks, list(checkpoint = cb))
  expect_identical(md$pointer, pointer)
  expect_equal(md$pointer_shape, pointer_shape)

  repr = capture.output(md)
  expected = c(
    "<ModelDescriptor: 2 ops>",
    "* Ingress:  torch_ingress_num.input: [(NA,4)]",
    "* Task:  iris [classif]",
    "* Callbacks:  Checkpoint",
    "* Optimizer:  Adaptive Moment Estimation",
    "* Loss:  Cross Entropy",
    "* pointer:  nn_linear.output [(NA,3)]"
  )

  expect_equal(expected, repr)

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

test_that("model_descriptor_union basic checks", {
  task = tsk("iris")
  task_sepal = task$clone()$select(c("Sepal.Length", "Sepal.Width"))
  task_petal = task$clone()$select(c("Petal.Length", "Petal.Width"))

  g1 = po("torch_ingress_num_1") %>>%
    po("nn_relu") %>>%
    po("nn_linear_1", out_features = 1) %>>%
    po("torch_loss", "cross_entropy") %>>%
    po("torch_callbacks_1", "progress")

  g2 = po("torch_ingress_num_2") %>>%
    po("nn_linear_2", out_features = 1) %>>%
    po("torch_optimizer", "sgd") %>>%
    po("torch_callbacks_2", "history")

  md_sepal = g1$train(task_sepal)[[1L]]
  md_petal = g2$train(task_petal)[[1L]]

  md = model_descriptor_union(md_sepal, md_petal)

  expect_equal(md$optimizer$id, "sgd")
  expect_equal(md$loss$id, "cross_entropy")
  expect_set_equal(ids(md$callbacks), c("progress", "history"))
  expect_true(length(md$callbacks) == 2)
  expect_set_equal(md$task$feature_names, task$feature_names)

  expected = setdiff(union(names(g1$pipeops), names(g2$pipeops)),
    c("torch_callbacks_2", "torch_loss", "torch_optimizer", "torch_callbacks_1")
  )
  expect_set_equal(expected, names(md$graph$pipeops))

  expected = rbindlist(list(
    list("torch_ingress_num_1", "output", "nn_relu", "input"),
    list("nn_relu", "output", "nn_linear_1", "input"),
    list("torch_ingress_num_2", "output", "nn_linear_2", "input")
  ))
  names(expected) = c("src_id", "src_channel", "dst_id", "dst_channel")
  setkeyv(expected, names(expected))
  observed = setkeyv(md$graph$edges, names(expected))
  expect_identical(observed, expected)

  md1 = po("torch_ingress_num")$train(list(task))[[1L]]
  md2 = model_descriptor_union(md1, md1)
  expect_identical(address(md1$graph), address(md2$graph))

  # not we check that we can add edges
  task = tsk("iris")
  graph = list(
    po("select_1", selector = selector_grep("Sepal")) %>>% po("torch_ingress_num_1") %>>% po("nn_linear_1", out_features = 1), # nolint
    po("select_2", selector = selector_grep("Petal")) %>>% po("torch_ingress_num_2") %>>% po("nn_linear_2", out_features = 2) # nolint
  ) %>>%
    po("nn_merge_sum", innum = 2)

  md = graph$train(task)[[1L]]

  expected_pos = c(
    "torch_ingress_num_1",
    "nn_linear_1",
    "torch_ingress_num_2",
    "nn_linear_2",
    "nn_merge_sum"
  )
  expect_identical(sort(names(md$graph$pipeops)), sort(expected_pos))

  expected_edges = rbindlist(list(
    list("torch_ingress_num_1", "output", "nn_linear_1", "input"),
    list("torch_ingress_num_2", "output", "nn_linear_2", "input"),
    list("nn_linear_1", "output", "nn_merge_sum", "input1"),
    list("nn_linear_2", "output", "nn_merge_sum", "input2")
  ))

  colnames(expected_edges) = c("src_id", "src_channel", "dst_id", "dst_channel")
  expected_edges = setkeyv(expected_edges, colnames(expected_edges))

  observed_edges = setkeyv(md$graph$edges, colnames(expected_edges))
  expect_identical(observed_edges, expected_edges)
})

test_that("model_descriptor_union verifies input correctly", {
  expect_error(model_descriptor_union(list(), list()), regexp = "Must inherit from class")

  task = tsk("iris")
  task_sepal = task$clone()$select(c("Sepal.Length", "Sepal.Width"))
  task_petal = task$clone()$select(c("Petal.Length", "Petal.Width"))

  md_sepal = po("torch_ingress_num")$train(list(task_sepal))[[1L]]
  md_petal = po("torch_ingress_num")$train(list(task_petal))[[1L]]

  expect_error(model_descriptor_union(md_sepal, md_petal), regexp = "Both graphs have")

  po_linear = po("nn_linear", out_features = 1)
  md1 = (po("torch_ingress_num_1") %>>% po_linear)$train(task)[[1L]]
  md2 = (po("torch_ingress_num_2") %>>% po_linear)$train(task)[[1L]]

  tmp = md2$graph$pipeops["nn_linear"]
  md2$graph$pipeops["nn_linear"] = md1$graph$pipeops["nn_linear"]
  expect_error(model_descriptor_union(md1, md2), regexp = "have differing incoming edges")
  md2$graph$pipeops["nn_linear"] = tmp

  md3 = po("torch_ingress_num")$train(list(task))[[1L]]

  task1 = tsk("german_credit")
  md5 = (po("select", selector = selector_type("integer")) %>>%
    po("torch_ingress_num"))$train(task1)[[1]]

  po_ce = po("torch_loss", "cross_entropy")

  mdce1 = po_ce$train(list(md5))[[1L]]
  mdce2 = po_ce$train(list(md5))[[1L]]

  expect_error(model_descriptor_union(mdce1, mdce2), regexp = "loss of two ModelDescriptors being merged")
  mdce2$loss = mdce1$loss
  expect_error(model_descriptor_union(mdce1, mdce2), regexp = NA)


  po_adam = po("torch_optimizer")
  mdadam1 = po_adam$train(list(md5))[[1L]]
  mdadam2 = po_adam$train(list(md5))[[1L]]

  expect_error(model_descriptor_union(mdadam1, mdadam2), regexp = "optimizer of two ModelDescriptors being merged")
  mdadam2$optimizer = mdadam1$optimizer
  expect_error(model_descriptor_union(mdadam1, mdadam2), regexp = NA)

  # the callbacks
  po_progress = po("torch_callbacks", "progress")

  mdprog1 = po_progress$train(list(md5))[[1L]]
  mdprog2 = po_progress$train(list(md5))[[1L]]

  expect_error(model_descriptor_union(mdprog1, mdprog2), regexp = "progress are identical")

  mdprog2$callbacks = mdprog1$callbacks
  expect_error(model_descriptor_union(mdprog1, mdprog2), regexp = NA)
})
