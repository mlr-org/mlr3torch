# When calling this function, everything is alrady prepared
train_eval = function(learner, history, epochs, callbacks, train_loader,
  valid_loader, context, ids
) {

  call = function(step) {
    call_back(step, callbacks, context)
  }

  call("on_start")

  for (epoch in seq_len(epochs)) {
    learner$network$train()

    call("on_before_train_epoch")

    loop(for (batch in train_loader) {

      learner$optimizer$zero_grad()

      call("on_before_train_batch")

      y_hat = learner$network$forward(batch$x)

      loss = learner$loss_fn(y_hat, batch$y)

      loss$backward()

      learner$optimizer$step()

      history$append("train_loss", loss$item(), "train")

      context$last = list(
        response = as.array(y_hat),
        truth = as.array(batch$y)
      )

      score_measures(context, "train")

      call("on_after_train_batch")

      history$increment("train")
    })

    learner$network$eval()

    call("on_before_valid_epoch")

    loop(for (batch in valid_loader) {

      call("on_before_valid_batch")

      y_hat = with_no_grad(learner$network$forward(batch$x))

      context$last = list(response = as.array(y_hat), truth = as.array(batch$y))

      score_measures(context, "valid")

      call("on_after_valid_batch")

      history$increment("valid")
    })

    call("on_after_valid_epoch")

    history$increment("epoch")

  }

  call("on_end")

  list(
    network = learner$network,
    optimizer = learner$optimizer,
    loss_fn = learner$loss_fn,
    history = history,
    ids = ids
  )
}


score_measures = function(context, phase) {
  measures = context$measures[[phase]]
  truth = context$last$truth
  response = context$last$response
  scores = imap(
    measures,
    function(measure, name) {
      measure(truth = truth, response = response)
    }
  )

  iwalk(
    scores,
    function(value, measure) {
      context$history$append(
        measure = measure,
        value = value,
        phase = phase
      )
    }
  )
}

learner_torch_train = function(self, model, task) {
  p = self$param_set$get_values(tags = "train")
  model$network$to(device = p$device)
  torch_set_num_threads(p$num_threads)


  ids = rsmp("holdout", ratio = 1 - p$valid_split)$instantiate(task)$instance
  names(ids) = c("train", "valid")

  train_set = as_dataset(task, device = p$device, augmentation = NULL,
    row_ids = ids$train
  )

  valid_set = as_dataset(task, device = p$device, augmentation = NULL,
    row_ids = ids$valid
  )

  train_loader = as_dataloader(train_set, batch_size = p$batch_size, drop_last = p$drop_last,
    shuffle = p$shuffle
  )

  valid_loader = as_dataloader(valid_set, batch_size = p$batch_size, drop_last = p$drop_last,
    shuffle = FALSE
  )


  # we set the state of the learner here for convenience, it will be reset to NULL before
  # returning from .train

  measures = split_list(p$measures, c("train|^$", "valid|^$"))
  measures = map(measures, get_measures)

  self$state = list(model = model)

  history = History$new(length(train_loader), length(valid_loader))
  self$state$history = history

  context = ContextTorch$new(
    learner = self,
    history = history,
    task = task,
    measures = measures
  )


  train_eval(
    learner = self,
    history = history,
    epochs = p$epochs,
    train_loader = train_loader,
    valid_loader = valid_loader,
    context = context,
    callbacks = p$callbacks,
    ids = ids
  )
}



build_torch = function(self, task, network = NULL) {
  p = self$param_set$get_values(tag = "train")

  pars_optim = p[startsWith(names(p), "opt.")]
  names(pars_optim) = gsub("opt.", "", names(pars_optim), fixed = TRUE)

  pars_loss = p[startsWith(names(p), "loss.")]
  names(pars_loss) = gsub("loss.", "", names(pars_loss), fixed = TRUE)

  pars = remove_named(p, c(names(pars_optim), names(pars_loss)))

  optim_name = get_private(self)$.optimizer
  loss_name = get_private(self)$.loss

  optimizer = invoke(get_optimizer(optim_name), .args = pars_optim, params = network$parameters)
  loss_fn = invoke(get_loss(loss_name), .args = pars_loss)

  list(
    network = network,
    optimizer = optimizer,
    loss_fn = loss_fn
  )
}
