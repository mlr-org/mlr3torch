train_eval = function(learner, network, optimizer, loss_fn, history, epochs, callbacks, train_loader,
  valid_loader, context, task, measures_train, measures_valid) {
  call = function(step) {
    call_back(step, callbacks, context)
  }

  call("on_start")

  for (epoch in seq_len(epochs)) {
    context$epoch = context$epoch + 1L

    network$train()

    history$train_loss[[context$epoch]] = numeric(length(train_loader))
    predictions = vector("list", length(train_loader))

    context$iter = 0L

    call("on_before_train_epoch")

    loop(for (batch in train_loader) {
      context$iter = context$iter + 1L
      optimizer$zero_grad()

      call("on_before_train_batch")

      y_hat = network(batch$x)

      loss = loss_fn(y_hat, batch$y)

      loss$backward()
      history$train_loss[[context$epoch]][[context$iter]] = loss$item()
      predictions[[context$iter]] = y_hat$detach()

      call("on_after_backward")

      optimizer$step()

      call("on_after_train_batch")
    })

    eval_network(learner, network = network, valid_loader, task, context, measures_valid)

    # TODO: implement early stopping
  }

  call("on_end")

  list(
    network = network,
    optimizer = optimizer,
    loss_fn = loss_fn,
    history = history,
    raw = NULL
  )
}


eval_network = function(learner, network, valid_loader, task, context, measures) {
  history = context$history
  network$eval()
  predictions = vector("list", length(valid_loader))
  iter = 1L
  if (length(valid_loader)) {
    loop(for (batch in valid_loader) {
      predictions[[iter]] = with_no_grad(network$forward(batch$x))
      iter = iter + 1L
    })

    prediction = torch_cat(predictions, dim = 1L)

    prediction = encode_prediction(learner, prediction, task)
    prediction = as_prediction_data(prediction, task = task, check = TRUE,
      row_ids = task$row_roles$early_stopping
    )
    prediction = as_prediction(prediction, task = task)

    walk(
      measures,
      function(measure) {
        score = measure$score(prediction, task = task, train_set = task$row_roles$use)
        history$valid[[measure$id]][[context$epoch]] = score
      }
    )
  }
  NULL
}


learner_torch_train = function(self, model, task) {
  p = self$param_set$get_values(tags = "train")
  model$network$to(device = p$device)
  torch_set_num_threads(p$num_threads)

  p$measures_valid = p$measures_valid %??% list()
  p$measures_train = p$measures_train %??% list()
  if (!is.list(p$measures_valid)) {
    p$measures_valid = list(p$measures_valid)
  }
  if (!is.list(p$measures_train)) {
    p$measures_train = list(p$measures_train)
  }

  train_set = as_dataset(task, device = p$device, augmentation = NULL,
    row_ids = task$row_roles$use
  )

  valid_set = as_dataset(task, device = p$device, augmentation = NULL,
    row_ids = task$row_roles$early_stopping
  )

  train_loader = as_dataloader(train_set, batch_size = p$batch_size, drop_last = p$drop_last,
    shuffle = p$shuffle
  )

  valid_loader = as_dataloader(valid_set, batch_size = p$batch_size, drop_last = p$drop_last,
    shuffle = FALSE
  )

  history = History$new(measures_valid = p$measures_valid, measures_train = p$measures_train)

  context = ContextTorch$new(
    learner = self,
    history = history,
    task = task,
    train_loader = train_loader,
    valid_loader = valid_loader
  )

  train_eval(
    learner = self,
    network = model$network,
    optimizer = model$optimizer,
    loss_fn = model$loss_fn,
    history = history,
    epochs = p$epochs,
    train_loader = train_loader,
    valid_loader = valid_loader,
    context = context,
    callbacks = p$callbacks,
    task = task,
    measures_valid = p$measures_valid,
    measures_train = p$measures_train
  )
}


