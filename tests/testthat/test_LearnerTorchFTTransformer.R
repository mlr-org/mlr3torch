test_that("nn_ft_transformer_block works", {
  ft_transformer = nn_ft_transformer(
    # d_hidden = 10,
    # d_block = 5,
    # dropout1 = 0.3,
    # dropout2 = 0.3
  )
  expect_class(resnet, "nn_ft_transformer_block")
  t = torch_randn(2, 5)
  tout = ft_transformer(t)
  # expect_equal(tout$shape, c(2, 5))
})

# TODO: separate tests for categorical and num. features? Or one big test that includes all?
test_that("PipeOpTorchFTTransformerBlock works", {
  po_ft_transformer = PipeOpTorchFTTransformerBlock$new()
  po_ft_transformer$param_set$set_values(
    # d_hidden_multiplier = 2, dropout1 = 0.5, dropout2 = 0.5
  )
  graph = po("torch_ingress_num") %>>%
    po_ft_transformer

  task = tsk("mtcars")
  net = graph$train(task)[[1L]]$graph

  t = torch_randn(2, 10)
  tout = net$train(t)
  expect_equal(tout[[1L]]$shape, c(2, 7))
})

test_that("LearnerTorchFTTransformer works", {
  learner = lrn("classif.tab_resnet", epochs = 1, batch_size = 50,
    n_blocks = 2, d_hidden = 10, dropout1 = 0.3, dropout2 = 0.3, d_block = 5
  )
  task = tsk("iris")
  learner$train(task)
  pred = learner$predict(task)
  expect_prediction(pred)
  expect_learner_torch(learner, task = tsk("iris"))
})
