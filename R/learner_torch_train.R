
learner_torch_train = function(self, private, super, task) {
  param_vals = self$param_set$get_values(tags = "train")

  torch_set_num_threads(param_vals$num_threads)

  normalize_to_list = function(x) {
    if (length(x) == 0) return(list())
    if (!is.list(x)) return(structure(list(x), names= x$id))
    if (is.null(names(x))) names(x) = map_chr(x, "id")
    x
  }

  dataset_train = task_dataset(
    task,
    feature_ingress_tokens = feature_ingress_tokens,
    target_batchgetter = target_batchgetter,
    device = param_vals$device
  )
  loader_train = dataloader(
    dataset = dataset_train,
    batch_size = param_vals$batch_size,
    drop_last = param_vals$drop_last,
    shuffle = param_vals$shuffle
  )

  task_valid = task$clone()$filter(integer(0))
  task_valid$set_row_roles(task$row_roles$early_stopping, "use")
  dataset_valid = if (task_valid$nrow) task_dataset(
    task_valid,
    feature_ingress_tokens = feature_ingress_tokens,
    target_batchgetter = target_batchgetter,
    device = param_vals$device
  )
  loader_valid = if (task_valid$nrow) dataloader(
    dataset = dataset_valid,
    batch_size = param_vals$batch_size,
    drop_last = param_vals$drop_last,
    shuffle = FALSE
  )

  network = private$.network(task)$network$to(device = param_vals$device)

  state = TorchState$new(
    learner = self,
    task_train = task,
    task_valid = if (task_valid$nrow) valid_task,
    loader_train = loader_train,
    loader_valid = loader_valid,
    measures_train = normalize_to_list(param_vals$measures_train),
    measures_valid = if (task_valid$nrow) normalize_to_list(param_vals$measures_valid) else list(),
    network = network,
    optimizer = private$.optimizer$get_optimizer(network$parameters),
    loss_fn = private$.loss$get_loss()
  )

  cbs = lapply(param_vals$callbacks, function(x) x$new(state))

  train_loop(state, cbs)
}

train_loop = function(state, cbs) {

  call = function(step_name) {
    lapply(cbs, function(x) x[[step]]())
  }

  ## we do this so if the learner should crash the intermediate progress is saved somewhere
  state$learner$model = list(
    network = state$network,
    optimizer = state$optimizer,
    loss_fn = state$loss_fn,
    callback_instances = cbs
  )

  # note that task_valid may be present (callbacks could do their own validation)
  does_validation = !is.null(state$measures_valid)

  call("on_begin")

  on.exit({
    # in case a callback wants to finalize things
    call("on_end")
  })

  state$network$train()

  state$epoch = 0
  while (state$epoch < state$total_epochs) {
    state$epoch = state$epoch + 1
    call("on_epoch_begin")



    call("on_before_train_epoch")

    predictions = list()
    state$batch = 0
    loop(for (batch in state$loader_train) {
      state$batch = state$batch + 1

      state$optimizer$zero_grad()

      call("on_batch_begin")

      y_hat = do.call(state$network, batch$x)

      loss = state$loss_fn(y_hat, batch$y)

      loss$backward()

      call("on_after_backward")

      state$last_loss = loss$item()
      predictions[[iter]] = y_hat$detach()

      optimizer$step()

      call("on_batch_end")
    })

    state$last_scores_train = measure_prediction(torch_cat(predictions, dim = 1L), state$measures_train, state$task_train)

    call("on_before_validation")
    if (does_validation) {
      state$network$eval()
      pred_tensor = torch_network_predict(state$network, state$loader_valid, call)
      state$last_scores_valid = measure_prediction(pred_tensor, state$measures_valid, state$task_valid)
      state$network$train()
    }
    call("on_epoch_end")
  }

  list(
    network = state$network,
    optimizer = state$optimizer,
    loss_fn = state$loss_fn,
    callback_instances = cbs
  )
}

torch_network_predict = function(network, loader, callback_receiver = function(step_name) NULL) {
  iter = 1L
  predictions = list()
  loop(for (batch in loader) {
    callback_receiver("on_batch_valid_begin")
    predictions[[iter]] = with_no_grad(do.call(network$forward, batch$x))
    iter = iter + 1
    callback_receiver("on_batch_valid_end")
  })
  torch_cat(predictions, dim = 1L)
}

encode_prediction = function(predict_tensor, predict_type, task) {
  response = prob = NULL
  if (task$task_type == "classif") {
    if (predict_type == "response") {
      response = as.integer(predict_tensor$argmax(dim = 2L))
      class(response) = "factor"
      levels(response) = task$levels(cols = task$target_names)[[1L]]
    } else if (predict_type == "prob") {
      prob = as.numeric(nnf_softmax(predict_tensor, dim = 2L))
      colnames(prob) = task$target_names
    }
    return(list(response = response, prob = prob))
  } else if (task$task_type == "regr") {
    if (predict_type == "response") {
      return(response = as.numeric(predict_tensor))
    } else {
      stopf("Invalid predict_type for task_type 'regr'.")
    }
  } else {
    stopf("Invalid task_type.")
  }

}


measure_prediction = function(pred_tensor, measures, task) {

  if (!length(measures)) return(structure(list(), names = character(0)))

  prediction = encode_prediction("prob", pred_tensor, task)
  prediction = as_prediction_data(prediction, task = task, check = TRUE,
    row_ids = task$row_roles$early_stopping
  )
  prediction = as_prediction(prediction, task = task)

  lapply(
    measures,
    function(measure) {
      measure$score(prediction, task = task, train_set = task$row_roles$use)
    }
  )
}
