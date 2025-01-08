test_that("autotest", {
  cb = t_clbk("lr_scheduler_cosine_annealing", T_max = 1)
  expect_torch_callback(cb)
})

test_that("decay works") {
  cb = t_clbk("lr_scheduler_decay")
  task = tsk("iris")
  n_epochs = 10

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )
  mlp$param_set$set_values(cb.lr_scheduler.T_max = 10)

  mlp$train(task)

  # check the lr at the end
}

test_that("custom LR scheduler works", {
    # first: just see if you can train a network
  cb = t_clbk("lr_scheduler_cosine_annealing")
  task = tsk("iris")
  n_epochs = 10
  set.seed(1)
  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )
  mlp$param_set$set_values(cb.lr_scheduler.T_max = 10)

  mlp$train(task)

  # check the lr at the end
})