test_that("autotest", {
  cb = t_clbk("lr_scheduler")
  expect_torch_callback(cb)
})

# for each lr scheduler
test_that("cosine annealing works") {
  # first: just see if you can train a network
  cb = t_clbk("lr_scheduler_cosine_annealing")
  task = tsk("iris")
  n_epochs = 10
  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )
  mlp$param_set$set_values(cb.lr_scheduler_cosine_annealing.T_max = 10)

  mlp$train(task)

  # set a random seed

  # train a network with just `torch` and the lr scheduler in question

  # the sequence of learning rates here is the expected values

  # check that the learning rates set by the callback (i.e. the optimizer's learning rate) match this
}