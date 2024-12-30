test_that("autotest", {
  cb = t_clbk("lr_scheduler")
  expect_torch_callback(cb)
})

# begin LLM
# TODO: decide whether to set random seeds
# may not be necessary if we only look at learning rates

# Helper to create identical networks for comparison
setup_networks = function() {
  # Raw torch network
  torch_net = nn_sequential(
    nn_linear(4, 10),
    nn_relu(),
    nn_linear(10, 3)
  )
  
  # MLR3 network
  mlr_net = lrn("classif.mlp",
    epochs = 10,
    batch_size = 150,
    neurons = 10,
    learning_rate = 0.1
  )
  
  list(torch = torch_net, mlr = mlr_net)
}

test_that("cosine annealing scheduler works", {
  nets = setup_networks()
  task = tsk("iris")
  
  # Setup raw torch
  opt_torch = optim_adam(nets$torch$parameters, lr = 0.1)
  scheduler_torch = lr_scheduler_cosine_annealing_lr(opt_torch, T_max = 10, eta_min = 0.001)
  
  # Setup mlr3torch
  nets$mlr$add_callback("lr_scheduler_cosine_annealing",
    T_max = 10,
    eta_min = 0.001
  )
  
  # Train both and collect learning rates
  lrs_torch = numeric()
  lrs_mlr = numeric()
  
  # Train for a few steps and compare learning rates
  for (i in 1:10) {
    scheduler_torch$step()
    lrs_torch[i] = scheduler_torch$get_last_lr()[[1]]
    
    if (i == 1) nets$mlr$train(task)  # This will create the scheduler
    nets$mlr$model$scheduler$step()
    lrs_mlr[i] = nets$mlr$model$optimizer$param_groups[[1]]$lr
  }
  
  expect_equal(lrs_torch, lrs_mlr)
})

test_that("step scheduler works", {
  nets = setup_networks()
  task = tsk("iris")
  
  # Setup raw torch
  opt_torch = optim_adam(nets$torch$parameters, lr = 0.1)
  scheduler_torch = lr_scheduler_step_lr(opt_torch, step_size = 3, gamma = 0.1)
  
  # Setup mlr3torch
  nets$mlr$add_callback("lr_scheduler_step",
    step_size = 3,
    gamma = 0.1
  )
  
  # Train both and collect learning rates
  lrs_torch = numeric()
  lrs_mlr = numeric()
  
  for (i in 1:10) {
    scheduler_torch$step()
    lrs_torch[i] = scheduler_torch$get_last_lr()[[1]]
    
    if (i == 1) nets$mlr$train(task)
    nets$mlr$model$scheduler$step()
    lrs_mlr[i] = nets$mlr$model$optimizer$param_groups[[1]]$lr
  }
  
  expect_equal(lrs_torch, lrs_mlr)
})

# Similar tests for other schedulers...
# end LLM

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