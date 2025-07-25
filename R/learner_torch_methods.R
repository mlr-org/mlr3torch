normalize_to_list = function(x) {
  if (length(x) == 0) {
    return(list())
  }
  if (!is.list(x)) {
    return(structure(list(x), names = x$id))
  }
  if (anyNA(names2(x))) names(x) = map_chr(x, "id")
  x
}

learner_torch_predict = function(self, private, super, task, param_vals) {
  # parameter like device "auto" already resolved
  self$network$to(device = param_vals$device)
  self$network$eval()
  data_loader = private$.dataloader_predict(private$.dataset(task, param_vals), param_vals)
  predict_tensor = torch_network_predict(self$network, data_loader, device = param_vals$device)
  private$.encode_prediction(predict_tensor = predict_tensor, task = task)
}

learner_torch_train = function(self, private, super, task, param_vals) {
  # Here, all param_vals (like seed = "random" or device = "auto") have already been resolved
  dataset_train = private$.dataset(task, param_vals)
  dataset_train = as_multi_tensor_dataset(dataset_train, param_vals)
  loader_train = private$.dataloader(dataset_train, param_vals)
  if (!length(loader_train)) {
    stopf("Training Dataloader of Learner '%s' has length 0", self$id)
  }

  network = private$.network(task, param_vals)
  network$to(device = param_vals$device)
  if (isTRUE(param_vals$jit_trace) && !inherits(network, "script_module")) {
    example = get_example_batch(loader_train)$x
    example = lapply(example, function(x) x$to(device = param_vals$device))
    # tracer requires arguments to be passed by name
    if (length(example) == 1) {
      network = jit_trace(network, example[[1L]])
    } else {
      example = order_named_args(network, example)
      network = do.call(jit_trace, c(list(network), unname(example)))
    }
  }
  if (is.null(self$optimizer)) stopf("Learner '%s' defines no optimizer", self$id)
  optimizer = self$optimizer$generate(network$parameters)
  if (is.null(self$loss)) stopf("Learner '%s' defines no loss", self$id)
  loss_fn = self$loss$generate(task)
  loss_fn$to(device = param_vals$device)

  measures_train = normalize_to_list(param_vals$measures_train)
  measures_valid = normalize_to_list(param_vals$measures_valid)

  if (length(measures_valid) && is.null(self$validate)) {
    stopf("Learner '%s' has measures_valid set, but its validate field is NULL. If this is happening in an AutoTuner, you probably forgot to set epochs = to_tune(upper = <max_epochs>, internal = TRUE).", self$id) # nolint
  }
  if (!length(measures_valid) && param_vals$patience != 0) {
    stopf("Learner '%s' has a non 0 patience parameter but has no measures_valid set.", self$id)
  }

  if (param_vals$patience > 0 && is.na(measures_valid[[1L]]$minimize)) {
    stopf("Learner '%s' uses a validation measure with minimize = NA for early stopping.", self$id)
  }

  task_valid = task$internal_valid_task
  loader_valid = if (!is.null(task_valid) && task_valid$nrow) {
    dataset_valid = private$.dataset(task_valid, param_vals)
    dataset_valid = as_multi_tensor_dataset(dataset_valid, param_vals)
    private$.dataloader_predict(dataset_valid, param_vals)
  }

  if (!is.null(loader_valid) && !length(loader_valid)) {
    stopf("Validation Dataloader of Learner '%s' has length 0", self$id)
  }

  ctx = ContextTorch$new(
    learner = self,
    task_train = task,
    task_valid = task_valid,
    loader_train = loader_train,
    loader_valid = loader_valid,
    measures_train = measures_train,
    measures_valid = measures_valid,
    network = network,
    optimizer = optimizer,
    loss_fn = loss_fn,
    total_epochs = param_vals$epochs,
    prediction_encoder = private$.encode_prediction,
    eval_freq = param_vals$eval_freq,
    device = param_vals$device
  )

  callbacks = set_names(lapply(self$callbacks, function(descriptor) {
    cb = descriptor$generate()
    cb$ctx = ctx
    cb
  }), ids(self$callbacks))


  if (param_vals$patience > 0L) {
    es = CallbackSetEarlyStopping$new(
      patience = param_vals$patience,
      min_delta = param_vals$min_delta
    )
    es$ctx = ctx

    callbacks = c(callbacks, list(early_stopping = es))
  }

  model = train_loop(ctx, callbacks)

  # In case the seed was "random" initially we want to make the sampled seed available in the state.
  model$seed = param_vals$seed

  structure(model, class = c("learner_torch_model", "list"))
}


train_loop = function(ctx, cbs) {
  call = function(step_name) {
    lapply(cbs, function(x) {
      if (exists(step_name, x, inherits = FALSE)) {
        x[[step_name]]()
      }
    })
  }

  # note that task_valid may be present (callbacks could do their own validation)
  on.exit({
    # in case a callback wants to finalize things
    call("on_exit")
    walk(cbs, function(cb) cb$ctx = NULL)
  }, add = TRUE)


  call("on_begin")

  ctx$network$train()

  # if we increment epoch at the end of the loop it has the wrong value
  # during the final two callback stages
  ctx$epoch = 0L
  while (ctx$epoch < ctx$total_epochs) {
    ctx$epoch = ctx$epoch + 1
    call("on_epoch_begin")

    predictions = list()
    indices = list()
    train_iterator = dataloader_make_iter(ctx$loader_train)
    ctx$step = 0L
    eval_train = eval_train_in_epoch(ctx)
    while (ctx$step < length(ctx$loader_train)) {
      ctx$step = ctx$step + 1
      ctx$batch = dataloader_next(train_iterator)
      ctx$batch$x = lapply(ctx$batch$x, function(x) x$to(device = ctx$device))
      ctx$batch$y = ctx$batch$y$to(device = ctx$device)
      ctx$optimizer$zero_grad()

      call("on_batch_begin")

      if (length(ctx$batch$x) == 1L) {
        ctx$y_hat = ctx$network(ctx$batch$x[[1L]])
      } else {
        ctx$y_hat = do.call(ctx$network, ctx$batch$x)
      }

      loss = ctx$loss_fn(ctx$y_hat, ctx$batch$y)

      loss$backward()

      call("on_after_backward")

      ctx$last_loss = loss$item()
      if (eval_train) {
        predictions[[length(predictions) + 1]] = ctx$y_hat$detach()
        indices[[length(indices) + 1]] = as.integer(ctx$batch$.index$to(device = "cpu"))
      }
      ctx$optimizer$step()

      call("on_batch_end")
    }

    ctx$last_scores_train = if (eval_train) {
      measure_prediction(
        pred_tensor = torch_cat(predictions, dim = 1L),
        measures = ctx$measures_train,
        task = ctx$task_train,
        row_ids = ctx$task_train$row_ids[unlist(indices)],
        prediction_encoder = ctx$prediction_encoder
      )
    }

    call("on_before_valid")
    if (eval_valid_in_epoch(ctx)) {
      ctx$network$eval()
      pred_tensor = torch_network_predict_valid(ctx, call)
      ctx$last_scores_valid = measure_prediction(
        pred_tensor = pred_tensor,
        measures = ctx$measures_valid,
        task = ctx$task_valid,
        row_ids = ctx$task_valid$row_ids,
        prediction_encoder = ctx$prediction_encoder
      )
      ctx$network$train()
      call("on_valid_end")
    } else {
      ctx$last_scores_valid = NULL
    }
    call("on_epoch_end")

    if (isTRUE(ctx$terminate)) break
  }

  call("on_end")

  callback_states = discard(map(cbs, function(cb) cb$state_dict()), is.null)
  # The seed is added later
  list(
    network               = ctx$network,
    # last epoch always does validation so this is fine
    internal_valid_scores = if (length(ctx$measures_valid)) ctx$last_scores_valid,
    loss_fn               = ctx$loss_fn$state_dict(),
    optimizer             = ctx$optimizer$state_dict(),
    epochs                = ctx$epoch,
    callbacks             = callback_states
  )
}

eval_train_in_epoch = function(ctx) {
  length(ctx$measures_train) && (!(ctx$epoch %% ctx$eval_freq) || ctx$epoch == ctx$total_epochs)
}
eval_valid_in_epoch = function(ctx) {
  !is.null(ctx$loader_valid) && (!(ctx$epoch %% ctx$eval_freq) || ctx$epoch == ctx$total_epochs)
}

has_one_arg = function(network) {
  if (inherits(network, "nn_graph")) {
    return(length(network$input_map) == 1L)
  }
  fargs = formalArgs(network)
  length(fargs) == 1L && !fargs == "..."
}

torch_network_predict_valid = function(ctx, callback_receiver = function(step_name) NULL) {
  network = ctx$network
  loader = ctx$loader_valid
  one_arg = has_one_arg(network)
  predictions = vector("list", length = length(loader))
  valid_iterator = dataloader_make_iter(loader)
  ctx$step_valid = 0L
  while (ctx$step_valid < length(loader)) {
    ctx$step_valid = ctx$step_valid + 1L
    ctx$batch = dataloader_next(valid_iterator)
    ctx$batch$x = lapply(ctx$batch$x, function(x) x$to(device = ctx$device))

    callback_receiver("on_batch_valid_begin")
    predictions[[ctx$step_valid]] = if (one_arg) {
      with_no_grad(network$forward(ctx$batch$x[[1L]]))
    } else {
      with_no_grad(invoke(network$forward, .args = ctx$batch$x))
    }

    callback_receiver("on_batch_valid_end")
  }
  torch_cat(predictions, dim = 1L)
}

torch_network_predict = function(network, loader, device) {
  # an unnamed argument
  # TODO: Maybe we should be stricter, but then we need to ensure that the .getbatch() method of the dataset
  # returns a list where the names of x correspond to the argument names of the network
  one_arg = has_one_arg(network)
  predictions = vector("list", length = length(loader))
  train_iterator = dataloader_make_iter(loader)
  step = 0L
  while (step < length(loader)) {
    step = step + 1L
    batch = dataloader_next(train_iterator)
    batch$x = lapply(batch$x, function(x) x$to(device = device))
    predictions[[step]] = if (one_arg) {
      with_no_grad(network$forward(batch$x[[1L]]))
    } else {
      with_no_grad(invoke(network$forward, .args = batch$x))
    }

  }
  torch_cat(predictions, dim = 1L)
}

encode_prediction_default = function(predict_tensor, predict_type, task) {
  # here we assume that the levels of the factors are never reordered!
  # This is important as otherwise all hell breaks loose
  # Currently this check is done in mlr3torch but should at some point be handled in mlr3 / mlr3pipelines

  response = prob = NULL
  if (task$task_type == "classif" && "multiclass" %in% task$properties) {
    if (predict_type == "prob") {
      predict_tensor = with_no_grad(nnf_softmax(predict_tensor, dim = 2L))
    }
    # We still execute the argmax on the device before converting to R
    response = as.integer(with_no_grad(predict_tensor$argmax(dim = 2L))$to(device = "cpu"))

    predict_tensor = predict_tensor$to(device = "cpu")
    prob = if (predict_type == "prob") {
      prob = as.matrix(predict_tensor)
      colnames(prob) = task$class_names
      prob
    }

    class(response) = "factor"
    levels(response) = task$class_names
    return(list(response = response, prob = prob))
  } else if (task$task_type == "classif") {
    # binary:
    # (first factor level is positive class)
    response = as.integer(with_no_grad(predict_tensor < 0)$to(device = "cpu") + 1)
    class(response) = "factor"
    levels(response) = task$class_names

    prob = if (predict_type == "prob") {
      # convert score to prob
      predict_tensor = with_no_grad(nnf_sigmoid(predict_tensor))
      prob = as.numeric(predict_tensor)
      prob = as.matrix(data.frame(prob, 1 - prob))
      colnames(prob) = task$class_names
      prob
    }

    return(list(response = response, prob = prob))
  } else if (task$task_type == "regr") {
    if (predict_type == "response") {
      return(list(response = as.numeric(predict_tensor)))
    } else {
      stopf("Invalid predict_type for task_type 'regr'.")
    }
  } else {
    stopf("Invalid task_type.")
  }
}


measure_prediction = function(pred_tensor, measures, task, row_ids, prediction_encoder) {
  if (!length(measures)) {
    return(structure(list(), names = character(0)))
  }

  prediction = prediction_encoder(predict_tensor = pred_tensor, task = task)
  prediction = as_prediction_data(prediction, task = task, check = FALSE, row_ids = row_ids)
  prediction = as_prediction(prediction, task = task, check = FALSE)

  lapply(
    measures,
    function(measure) {
      tryCatch(
        measure$score(prediction, task = task, train_set = task$row_roles$use),
        error = function(e) NaN
      )
    }
  )
}

as_multi_tensor_dataset = function(dataset, param_vals) {
 if (isTRUE(param_vals$tensor_dataset)) {
    multi_tensor_dataset(dataset, device = "cpu")
  } else if (identical(param_vals$tensor_dataset, "device")) {
    multi_tensor_dataset(dataset, device = param_vals$device)
  } else {
    dataset
  }
}
