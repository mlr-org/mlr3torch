test_that("Autotest", {
  cb = t_clbk("checkpoint", freq = 1, path = tempfile())
  expect_torch_callback(cb)
})

test_that("manual", {
  cb = t_clbk("checkpoint", freq = 1)
  task = tsk("iris")
  task$row_roles$use = 1

  pth0 = tempfile()
  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
  learner$param_set$set_values(cb.checkpoint.path = pth0)

  learner$train(task)

  expect_set_equal(
    c(paste0("network", 1:3, ".pt"), paste0("optimizer", 1:3, ".pt")),
    list.files(pth0)
  )


  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
  pth2 = tempfile()
  learner$param_set$set_values(cb.checkpoint.path = pth2, cb.checkpoint.freq = 2)
  learner$train(task)

  expect_set_equal(
    c("network2.pt", "optimizer2.pt", "network3.pt", "optimizer3.pt"),
    list.files(pth2)
  )
  pred = learner$predict(tsk("iris"))

  opt_state = torch_load(file.path(pth2, "optimizer3.pt"))
  expect_list(opt_state, types = c("numeric", "list", "torch_tensor"))
})

test_that("error when using existing directory", {
  path = tempfile()
  dir.create(path)
  cb = t_clbk("checkpoint", freq = 1, path = path)
  expect_error(cb$generate(), "already exists")
})
