test_that("autotest", {
  cb = t_clbk("lr_scheduler_cosine_annealing", T_max = 1)
  expect_torch_callback(cb)
})

test_that("cosine annealing works") {
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

  set.seed(1)
  torch_mlp <- nn_sequential(
    nn_flatten(),
    nn_linear(4, 10),
    nn_relu(),
    nn_linear(10, 3)
  )

  train_ds = torch::dataset("iris")

  opt <- optim_adam(torch_mlp$parameters)
  for (t in seq_len(n_epochs)) {
    coro::loop(for (b in train_dl) {
      opt$zero_grad()

      output = learner(b[[1]]$to(device = accelerator))
      target = b[[2]]$to(device = accelerator)
      loss = nnf_mse_loss(output$squeeze(2), target)

      loss$backward()
      opt$step()
    })
  }

  # set a random seed

  # train a network with just `torch` and the lr scheduler in question

  # the sequence of learning rates here is the expected values

  # check that the learning rates set by the callback (i.e. the optimizer's learning rate) match this

}

# Similar tests for other schedulers...
# end LLM

# for each lr scheduler
# test_that("cosine annealing works") {
#   # first: just see if you can train a network
#   cb = t_clbk("lr_scheduler_cosine_annealing")
#   task = tsk("iris")
#   n_epochs = 10
#   mlp = lrn("classif.mlp",
#             callbacks = cb,
#             epochs = n_epochs, batch_size = 150, neurons = 10,
#             measures_train = msrs(c("classif.acc", "classif.ce"))
#   )
#   mlp$param_set$set_values(cb.lr_scheduler_cosine_annealing.T_max = 10)
#
#   mlp$train(task)
#
#   # set a random seed
#
#   # train a network with just `torch` and the lr scheduler in question
#
#   # the sequence of learning rates here is the expected values
#
#   # check that the learning rates set by the callback (i.e. the optimizer's learning rate) match this
# }


