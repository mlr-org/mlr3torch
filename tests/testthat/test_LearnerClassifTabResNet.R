test_that("LearnerClassifTabResNet works", {
  task = tsk("iris")
  l = lrn("classif.tab_resnet",
    batch_size = 16L,
    epochs = 1L,
    n_blocks = 2L,
    d_hidden = 20L,
    d_main = 20L,
    dropout_first = 0.2,
    dropout_second = 0.3,
    activation = "relu",
    skip_connection = TRUE
  )
  expect_error(l$train(task), regexp = NA)
})
