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
    c(paste0("network", 1:3, ".pt"), paste0("optimizer", 1:3, ".pt"), "learner.rds", "loss.pt", "config.json"),
    list.files(pth0)
  )

  learner = readRDS(file.path(pth0, "learner.rds"))
  expect_learner(learner)

  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
  pth2 = tempfile()
  learner$param_set$set_values(cb.checkpoint.path = pth2, cb.checkpoint.freq = 2)
  learner$train(task)

  expect_set_equal(
    c("network2.pt", "optimizer2.pt", "learner.rds", "loss.pt", "config.json"),
    list.files(pth2)
  )
  learner = readRDS(file.path(pth2, "learner.rds"))
  expect_class(learner, "LearnerTorch")
  network_state = torch::torch_load(file.path(pth2, "network2.pt"))
  expect_list(network_state, types = c("numeric", "list", "torch_tensor"))
  learner$network$load_state_dict(network_state)
  pred = learner$predict(tsk("iris"))


  opt_state = torch_load(file.path(pth2, "optimizer3.pt"))
  expect_list(opt_state, types = c("numeric", "list", "torch_tensor"))

  # parameter is respected
  pth3 = tempfile()
  learner$param_set$set_values(cb.checkpoint.save = "network", cb.checkpoint.path = pth3)
  learner$train(task)

  expect_set_equal(
    paste0("network", 2:3, ".pt"),
    list.files(pth3)
  )

  pth4 = tempfile()
  learner$param_set$set_values(
    cb.checkpoint.save = "network",
    cb.checkpoint.path = pth4,
    cb.checkpoint.freq_type = "epoch",
    cb.checkpoint.freq = 1,
    epochs = 3
  )
  learner$train(task)

  # learner is also saved if wanted
})

test_that("can use existing directory for output", {
  path = tempfile()
  dir.create(path)
  cb = t_clbk("checkpoint", freq = 1, path = path)
  expect_silent(cb$generate())
})
