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
  expect_deep_clone(learner, learner$clone(deep = TRUE))
  expect_learner_torch(learner)
})
