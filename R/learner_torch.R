# Here are the standard methods that are shared between all the TorchLearners
learner_classif_torch_predict = function(self, task) {
  was_in_train_mode = self$state$model$network$training
  on.exit(if (was_in_train_mode) self$state$model$network$train(), add = TRUE)

  network = self$state$model$network

  pars = self$param_set$get_values(tags = "predict")
  device = pars$device
  batch_size = pars$batch_size

  data_loader = as_dataloader(task, device = device, batch_size = batch_size)
  npred = length(data_loader$dataset) # length of dataloader are the batches
  responses = integer(npred)
  i = 0L
  coro::loop(for (batch in data_loader) {
    p = with_no_grad(
      network$forward(batch$x)
    )
    p = as.integer(p$argmax(dim = 2L))
    # TODO: differentiate between different predict types
    responses[(i * batch_size + 1L):min(((i + 1L) * batch_size), npred)] = p
    i = i + 1L
  })

  responses = as.factor(responses)
  browser()
  levels(responses) = task$levels(cols = task$target_names)[[1L]]
  list(response = responses)
}

# Train function for the classification torch learne
learner_classif_torch_train = function(self, state, task) {

  pars = self$param_set$get_values(tags = "train")
  epochs = pars$epochs
  device = pars$device
  batch_size = pars$batch_size
  drop_last = pars$drop_last %??% FALSE
  shuffle = pars$shuffle %??% TRUE

  # TODO: "train" set is currently not really supprted (I think) must to setdiff(use, test) (?)
  warningf("train and valid loader are the same")
  train_loader = as_dataloader(task, device = device, batch_size = batch_size,
    shuffle = shuffle, drop_last = drop_last
  )
  valid_loader = as_dataloader(task, device = device, batch_size = batch_size,
    shuffle = shuffle, drop_last = drop_last
  )

  network = state$network
  optimizer = state$optimizer
  criterion = state$criterion
  history = state$history

  for (epoch in seq_len(epochs)) {
    losses = list()
    loop(for (batch in train_loader) {
      optimizer$zero_grad()
      y = batch$y
      x = batch$x
      y_hat = network$forward(x)
      y_true = batch$y[, 1L]

      loss = criterion(y_hat, y_true)
      loss$backward()
      optimizer$step()
      losses = c(losses, loss$item())
    })

    with_no_grad(
      loop(for (batch in valid_loader) {
        y = batch$y
        # x = batch4
      })
    )
  }
  list(
    network = network,
    optimizer = optimizer,
    criterion = criterion,
    history = history
  )
}




build_torch = function(self, task) {
  pars = self$param_set$get_values(tag = "train")
  if (test_r6(pars$architecture, "Architecture")) {
    network = pars$architecture$build(task)
  } else if (test_r6(pars$architecture, "nn_Module")) {
    network = pars$architecture$clone(deep = TRUE)
  } else {
    stopf("Invalid argument for architecture.")
  }

  optimizer_args = pars$optimizer_args %??% list()
  criterion_args = pars$criterion_args %??% list()
  optimizer = invoke(get_optimizer(pars$optimizer), .args = optimizer_args, params = network$parameters)
  criterion = invoke(get_criterion(pars$criterion), .args = criterion_args)

  list(
    network = network,
    optimizer = optimizer,
    criterion = criterion,
    history = History$new()
  )
}
