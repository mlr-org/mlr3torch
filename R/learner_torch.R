# Here are the standard methods that are shared between all the TorchLearners
learner_classif_torch_predict = function(self, task) {
  model = self$state$model
  reset_train = model$network$training
  on.exit(if (reset_train) model$network$train(), add = TRUE)
  model$network$eval()

  network = model$network

  pars = self$param_set$get_values(tags = "predict")
  device = pars$device
  batch_size = pars$batch_size

  data_loader = as_dataloader(task, device = device, batch_size = batch_size, drop_last = FALSE)
  npred = length(data_loader$dataset) # length of dataloader are the batches
  responses = integer(npred)
  i = 0L
  coro::loop(for (batch in data_loader) {
    p = with_no_grad(
      network$forward(batch$x)
    )
    p = as.integer(p$argmax(dim = 2L)$to(device = "cpu"))
    # TODO: differentiate between different predict types
    responses[(i * batch_size + 1L):min(((i + 1L) * batch_size), npred)] = p
    i = i + 1L
  })

  # TODO: Check that nothing goes wrong here
  class(responses) = "factor"
  levels(responses) = task$levels(cols = task$target_names)[[1L]]
  list(response = responses)
}

# Train function for the classification torch learne
learner_classif_torch_train = function(self, model, task) {
  c(network, optimizer, loss_fn, history) %<-% model[c("network", "optimizer", "loss_fn", "history")]
  reset_eval = network$training
  on.exit(if (reset_eval) network$eval(), add = TRUE)
  network$train()

  pars = self$param_set$get_values(tags = "train")
  epochs = pars$epochs
  device = pars$device
  batch_size = pars$batch_size
  drop_last = pars$drop_last %??% FALSE
  shuffle = pars$shuffle %??% TRUE
  valid_split = pars$valid_split
  augmentation = pars$augmentatoin

  train_fn = pars$train_fn
  valid_fn = pars$valid_fn

  valid_rsmp = rsmp("holdout", ratio = 1 - valid_split)

  torch_set_num_threads(pars$num_threads)

  c(train_ids, valid_ids) %<-% valid_rsmp$instantiate(task)$instance

  train_set = as_dataset(task, device, augmentation, train_ids)
  valid_set = as_dataset(task, device, NULL, valid_ids)

  train_loader = as_dataloader(train_set, batch_size = batch_size, drop_last = drop_last,
    augmentation = train_augmentation
  )
  valid_loader = as_dataloader(valid_set, batch_size = batch_size, drop_last = drop_last)

  history$n_train = length(train_loader)
  history$n_valid = length(valid_loader)

  for (epoch in seq_len(epochs)) {
    history$train_iter = 1L

    pb = progress::progress_bar$new(
      total = length(train_loader),
      format = "[:bar] :eta Loss: :loss"
    )

    loop(for (batch in train_loader) {
      train_fn(batch, network, optimizer, loss_fn, history)
      history$train_iter = history$train_iter + 1L
      pb$tick(tokens = list(loss = history$last_train_loss))
    })

    pb = progress::progress_bar$new( total = length(valid_loader), format = "[:bar]")

    history$valid_iter = 1L
    loop(for (batch in valid_loader) {
      valid_fn(batch, network, loss_fn, history)
      history$valid_iter = history$valid_iter + 1L
      pb$tick()
    })
    valid_loss = mean(history$valid_loss[[epoch]])
    train_loss = mean(history$train_loss[[epoch]])
    cat(sprintf("[epoch %d]: [Loss] Train = %3f, Valid = %3f \n", epoch, train_loss, valid_loss))

    history$epoch = history$epoch + 1L
  }
  list(
    network = network,
    optimizer = optimizer,
    loss_fn = loss_fn,
    history = history,
    valid_ids = valid_ids
  )
}

default_epoch_fn = function(network, optimizer, loss_fn, history) {
  NULL
}

default_train_fn = function(batch, network, optimizer, loss_fn, history) {
  optimizer$zero_grad()
  y_hat = network$forward(batch$x)
  loss = loss_fn(y_hat, batch$y)
  loss$backward()
  optimizer$step()
  history$add_train_loss(loss$item())
  NULL
}

default_valid_fn = function(batch, network, loss_fn, history) {
  y_hat = with_no_grad(network$forward(batch$x))
  loss = loss_fn(y_hat, batch$y)
  history$add_valid_loss(loss$item())
  NULL
}


build_torch = function(self, task, network = NULL) {
  pars = self$param_set$get_values(tag = "train")

  pars_optim = pars[startsWith(names(pars), "opt.")]
  names(pars_optim) = gsub("opt.", "", names(pars_optim), fixed = TRUE)

  pars_loss = pars[startsWith(names(pars), "loss.")]
  names(pars_loss) = gsub("loss.", "", names(pars_loss), fixed = TRUE)

  pars = remove_named(pars, c(names(pars_optim), names(pars_loss)))

  optim_name = get_private(self)$.optimizer
  loss_name = get_private(self)$.loss


  optimizer = invoke(get_optimizer(optim_name), .args = pars_optim, params = network$parameters)
  loss_fn = invoke(get_loss(loss_name), .args = pars_loss)

  list(
    network = network,
    optimizer = optimizer,
    loss_fn = loss_fn,
    history = History$new()
  )
}
