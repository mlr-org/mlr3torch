test_that("Autotest", {
  cb = t_clbk("checkpoint", freq = 1, path = tempfile())
  expect_torch_callback(cb)
})

test_that("CallbackSetCheckpoint manual", {
  cb = t_clbk("checkpoint", freq = 1)
  task = tsk("iris")
  task$row_roles$use = 1

  pth0 = tempfile()
  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
  learner$param_set$set_values(cb.checkpoint.path = pth0)

  learner$train(task)

  expect_set_equal(paste0("network", 1:3, ".pt"), list.files(pth0))

  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
  pth2 = tempfile()
  learner$param_set$set_values(cb.checkpoint.path = pth2, cb.checkpoint.freq = 2)
  learner$train(task)

  expect_set_equal(paste0("network", 2:3, ".pt"), list.files(pth2))
})
