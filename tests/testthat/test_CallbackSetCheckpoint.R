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
    c(paste0("network", 1:3, ".pt"), paste0("optimizer", 1:3, ".pt"), paste0("state", 1:3, ".rds")),
    list.files(pth0)
  )


  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
  pth2 = tempfile()
  learner$param_set$set_values(cb.checkpoint.path = pth2, cb.checkpoint.freq = 2)
  learner$train(task)

  expect_set_equal(
    c("network2.pt", "optimizer2.pt", "state2.rds", "network3.pt", "optimizer3.pt", "state3.rds"),
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

test_that("a folder that already contains checkpoints can be continued in", {
  task = tsk("iris")
  path = tempfile()
  learner = lrn("classif.mlp", epochs = 1, batch_size = 50, neurons = 10,
    callbacks = t_clbk("checkpoint", freq = 1))
  learner$param_set$set_values(cb.checkpoint.path = path)
  learner$train(task)

  # writing more checkpoints into the same folder is what happens when a run is resumed
  expect_class(t_clbk("checkpoint", freq = 1, path = path)$generate(), "CallbackSetCheckpoint")

  # a folder with unrelated content is still rejected
  other = tempfile()
  dir.create(other)
  writeLines("", file.path(other, "some_file.txt"))
  expect_error(t_clbk("checkpoint", freq = 1, path = other)$generate(), "already exists")
})

test_that("the state contains the epoch and the states of the other callbacks", {
  task = tsk("iris")
  pth = tempfile()
  learner = lrn("classif.mlp", epochs = 2, batch_size = 50, neurons = 10,
    measures_train = msrs("classif.acc"),
    callbacks = list(t_clbk("checkpoint", freq = 1), t_clbk("history")))
  learner$param_set$set_values(cb.checkpoint.path = pth)
  learner$train(task)

  state = readRDS(file.path(pth, "state2.rds"))
  expect_equal(state$epoch, 2L)
  expect_names(names(state$callbacks), identical.to = "history")
  # saveRDS() is used so that classes such as data.table survive, which torch_save() does not keep
  expect_data_table(state$callbacks$history, nrows = 2L)
  expect_equal(state$callbacks$history$epoch, 1:2)
})
