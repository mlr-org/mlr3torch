test_that("nn_tab_resnet_block works", {
  resnet = nn_tab_resnet_block(
    d_hidden = 10,
    d_block = 5,
    dropout1 = 0.3,
    dropout2 = 0.3
  )
  expect_class(resnet, "nn_tab_resnet_block")
  t = torch_randn(2, 5)
  tout = resnet(t)
  expect_equal(tout$shape, c(2, 5))
})

test_that("PipeOpTorchTabResnetBlock works", {
  po_resnet = PipeOpTorchTabResNetBlock$new()
  po_resnet$param_set$set_values(
    d_hidden_multiplier = 2, dropout1 = 0.5, dropout2 = 0.5
  )
  graph = po("torch_ingress_num") %>>%
    po("nn_linear", out_features = 7) %>>%
    po_resnet

  task = tsk("mtcars")
  net = graph$train(task)[[1L]]$graph

  t = torch_randn(2, 10)
  tout = net$train(t)
  expect_equal(tout[[1L]]$shape, c(2, 7))
})

test_that("LearnerTorchTabResNet works", {
  learner = lrn("classif.tab_resnet", epochs = 1, batch_size = 50,
    n_blocks = 2, d_hidden = 10, dropout1 = 0.3, dropout2 = 0.3, d_block = 5
  )
  task = tsk("iris")
  learner$train(task)
  pred = learner$predict(task)
  expect_prediction(pred)
  expect_learner_torch(learner, task = tsk("iris"))
})

test_that("cloning also keeps parameter values", {
  learner = lrn("classif.tab_resnet", n_blocks = 2)
  learnerc = learner$clone(deep = TRUE)
  expect_deep_clone_mlr3torch(learner, learnerc)
  expect_equal(learner$param_set$values$n_blocks, 2)
  expect_equal(learnerc$param_set$values$n_blocks, 2)
})

test_that("task types", {
  learner = lrn("classif.tab_resnet", n_blocks = 0, epochs = 0, batch_size = 16, d_block = 5, d_hidden = 10, dropout1 = 0.3, dropout2 = 0.3)
  expect_learner_torch(learner, tsk("iris"))
  expect_learner_torch(learner, tsk("sonar"))
  learner = lrn("regr.tab_resnet", n_blocks = 0, epochs = 0, batch_size = 16, d_block = 5, d_hidden = 10, dropout1 = 0.3, dropout2 = 0.3)
  expect_learner_torch(learner, tsk("mtcars"))
})


test_that("lazy tensor works", {
  learner = lrn("classif.tab_resnet", n_blocks = 1, epochs = 1, batch_size = 16, d_block = 5, d_hidden = 10, dropout1 = 0.3, dropout2 = 0.3)
  task = tsk("lazy_iris")$filter(c(1, 51))
  expect_error(learner$train(task), regexp = NA)
})


test_that("error messages", {
  l = lrn("classif.tab_resnet", n_blocks = 1, epochs = 0, batch_size = 16, d_block = 5, d_hidden = 10, dropout1 = 0.3, dropout2 = 0.3)
  expect_error(l$train(nano_imagenet()), "expects an input shape of length 2")
  expect_error(l$train(nano_dogs_vs_cats()), "Please specify the learner's `shape` parameter")
  l$configure(shape = c(NA, 4))
  expect_error(l$train(nano_dogs_vs_cats()), NA)
})

test_that("the output of the last block is normalized and activated before the head", {
  # the prediction is Linear(ReLU(BatchNorm(x))) in Gorishniy et al. (2021), as the blocks only
  # normalize their input
  for (n_blocks in 0:2) {
    learner = lrn("regr.tab_resnet", epochs = 1, batch_size = 16, n_blocks = n_blocks, d_block = 5,
      d_hidden = 10, dropout1 = 0, dropout2 = 0)
    learner$train(tsk("mtcars"))
    graph = learner$network$graph
    src_of = function(id) graph$edges$src_id[graph$edges$dst_id == id]

    expect_equal(src_of("nn_head"), "nn_relu")
    expect_equal(src_of("nn_relu"), "nn_batch_norm1d")
    expect_equal(src_of("nn_batch_norm1d"),
      if (n_blocks) sprintf("nn_block.nn_tab_resnet__%i", n_blocks) else "nn_linear")
    expect_class(graph$pipeops$nn_batch_norm1d$module, "nn_batch_norm1d")
    expect_equal(graph$pipeops$nn_batch_norm1d$module$num_features, 5)
  }
})
