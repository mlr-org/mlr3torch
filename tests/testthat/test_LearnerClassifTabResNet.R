test_that("LearnerClassifTabResNet works", {
  l = lrn("classif.tab_resnet",
    loss = "cross_entropy",
    optimizer = "adam",
    n_blocks = 2L,
    d_hidden = 30L,
    d_main = 30L,
    dropout_first = 0.2,
    dropout_second = 0.2,
    activation = "relu",
    activation_args = list(),
    skip_connection = TRUE,
    bn.momentum = 0.1,
    # training args
    batch_size = 16L,
    epochs = 10L,
    valid_split = 0.2,
    opt.lr = 0.03,
    callbacks = list(),
    shuffle = TRUE
  )

  result = run_autotest(learner = l, check_replicable = FALSE)

  expect_true(result, info = result$error)
})

if (FALSE) {
  task = tsk("iris")
  l$train(task)
}
