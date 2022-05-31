test_that("Callbacks work", {
  callbacks = list(
    cllb("torch.progress")
  )
  l = lrn("classif.alexnet",
    epochs = 1L,
    batch_size = 16L,
    device = "cpu",
    callbacks = callbacks,
    valid_split = 0.2,
    measures = list()
  )
  task = toytask()
  task$row_roles$use = 1:40
  l$train(task)
  expect_error(l$train(task), regexp = NA)
})
