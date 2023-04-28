test_that("Autotest", {
  cb = t_clbk("checkpoint")
  autotest_torch_callback(cb, list(freq = 1, path = tempfile()))
})

test_that("CallbackTorchCheckpoint manual", {
  cb = t_clbk("checkpoint", freq = 1)
  task = tsk("iris")
  task$row_roles$use = 1

  pth0 = tempfile()
  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, layers = 0, d_hidden = 1, callbacks = cb)
  learner$param_set$set_values(cb.checkpoint.path = pth0)

  learner$train(task)

  expect_set_equal(paste0("network", 1:3, ".pt"), list.files(pth0))

  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, layers = 0, d_hidden = 1, callbacks = cb)
  pth1 = tempfile()
  learner$param_set$set_values(cb.checkpoint.path = pth1, cb.checkpoint.save_last = FALSE, cb.checkpoint.freq = 2)
  learner$train(task)
  expect_set_equal(paste0("network", 2, ".pt"), list.files(pth1))


  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, layers = 0, d_hidden = 1, callbacks = cb)
  pth2 = tempfile()
  learner$param_set$set_values(cb.checkpoint.path = pth2, cb.checkpoint.save_last = TRUE, cb.checkpoint.freq = 2)
  learner$train(task)

  expect_set_equal(paste0("network", 2:3, ".pt"), list.files(pth2))
})
