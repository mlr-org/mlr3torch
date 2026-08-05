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

test_that("error when using an existing directory that holds unrelated data", {
  path = tempfile()
  dir.create(path)
  writeLines("not a checkpoint", file.path(path, "notes.txt"))
  cb = t_clbk("checkpoint", freq = 1, path = path)
  expect_error(cb$generate(), "already exists")
})

test_that("an existing empty directory can be checkpointed into", {
  # a pre-created output folder, and what a run that died before its first checkpoint leaves behind
  path = tempfile()
  dir.create(path)
  expect_no_error(t_clbk("checkpoint", freq = 1, path = path)$generate())

  task = tsk("iris")
  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("checkpoint", freq = 1, path = path))
  expect_no_error(learner$train(task))
  expect_set_equal(list.files(path),
    c(paste0("network", 1:2, ".pt"), paste0("optimizer", 1:2, ".pt")))
})

test_that("an epoch that was interrupted is not saved under its own number", {
  task = tsk("iris")
  path = tempfile()
  # crashes in the middle of epoch 3, after epoch 2 was saved by frequency
  crash = torch_callback("Crash",
    on_batch_end = function() if (self$ctx$epoch == 3L && self$ctx$step == 2L) stop("crash"))
  learner = lrn("classif.mlp", epochs = 6L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 2, path = path), crash))
  expect_error(learner$train(task), "crash")

  # 'network<n>.pt' is the network at the end of epoch n, and epoch 3 never reached its end
  expect_set_equal(list.files(path), c("network2.pt", "optimizer2.pt"))

  # the final epoch is still stored when training completes normally, also when `freq` skips it
  path2 = tempfile()
  done = lrn("classif.mlp", epochs = 3L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("checkpoint", freq = 2, path = path2))
  done$train(task)
  expect_set_equal(list.files(path2),
    c("network2.pt", "optimizer2.pt", "network3.pt", "optimizer3.pt"))
})
