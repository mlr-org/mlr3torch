test_that("Callbacks work", {
  callbacks = list(
    cllb("torch.progress")
  )
  l = lrn("classif.tab_resnet",
    epochs = 1L,
    batch_size = 16L,
    device = "cpu",
    callbacks = callbacks,
    measures = list(),
    optimizer = "adam",
    loss = "cross_entropy",
    n_blocks = 1,
    d_main = 10,
    d_hidden = 20,
    dropout_first = 0.2,
    dropout_second = 0.3
  )
  task = tsk("iris")
  l$train(task)
  expect_error(l$train(task), regexp = NA)
})
