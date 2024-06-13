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
  data_loader = private$.dataloader_predict(task, param_vals)
  predict_tensor = torch_network_predict(self$network, data_loader)
  private$.encode_prediction(predict_tensor = predict_tensor, task = task)
}

learner_torch_train = function(self, private, super, task, param_vals) {
  # Here, all param_vals (like seed = "random" or device = "auto") have already been resolved
  loader_train = private$.dataloader(task, param_vals)

  network = private$.network(task, param_vals)$to(device = param_vals$device)
  optimizer = private$.optimizer$generate(network$parameters)
  loss_fn = private$.loss$generate()

  measures_train = normalize_to_list(param_vals$measures_train)
  measures_valid = normalize_to_list(param_vals$measures_valid)

  task_valid = task$clone()$filter(integer(0))
  task_valid$set_row_roles(task$row_roles$test, "use")
  loader_valid = if (task_valid$nrow) {
    private$.dataloader_predict(task_valid, param_vals)
  } else {
    if (length(measures_valid)) {
      lg$warn("No validation set provided but measures for validation set specified.")
    }
    NULL
  }


  ctx = ContextTorch$new(
    learner = self,
    task_train = task,
    task_valid = if (task_valid$nrow) task_valid,
    loader_train = loader_train,
    loader_valid = loader_valid,
    measures_train = measures_train,
    measures_valid = measures_valid,
    network = network,
    optimizer = optimizer,
    loss_fn = loss_fn,
    total_epochs = param_vals$epochs,
    prediction_encoder = private$.encode_prediction
  )

  callbacks = lapply(private$.callbacks, function(descriptor) {
    cb = descriptor$generate()
    cb$ctx = ctx
    cb
  })

  model = train_loop(ctx, callbacks)

  # In case the seed was "random" initially we want to make the sampled seed available in the state.
  model$seed = param_vals$seed

  structure(model, class = c("learner_torch_state", "list"))
}


train_loop = function(ctx, cbs) {
  call = function(step_name) {
    lapply(cbs, function(x) {
      if (exists(step_name, x, inherits = FALSE)) {
        x[[step_name]]()
      }
    })
  }

  ## we do this so if the learner should crash the intermediate progress is saved somewhere
  ctx$learner$state$model = list(
    network = ctx$network,
    optimizer = ctx$optimizer,
    loss_fn = ctx$loss_fn,
    callbacks = cbs
  )

  # note that task_valid may be present (callbacks could do their own validation)
  does_validation = length(ctx$measures_valid) && !is.null(ctx$task_valid)

  on.exit({
    # in case a callback wants to finalize things
    call("on_exit")
    walk(cbs, function(cb) cb$ctx = NULL)
  }, add = TRUE)


  call("on_begin")

  ctx$network$train()

  ctx$epoch = 0
  while (ctx$epoch < ctx$total_epochs) {
    ctx$epoch = ctx$epoch + 1
    call("on_epoch_begin")

    predictions = list()
    indices = list()
    ctx$step = 0
    coro::loop(for (batch in ctx$loader_train) {
      ctx$step = ctx$step + 1

      ctx$optimizer$zero_grad()

      call("on_batch_begin")

      if (length(batch$x) == 1L) {
        # With one argument there is no ambiguity and we can be less strict
        # TODO: Make this more strict
        y_hat = ctx$network(batch$x[[1L]])
      } else {
        y_hat = do.call(ctx$network, batch$x)
      }

      loss = ctx$loss_fn(y_hat, batch$y)

      loss$backward()

      call("on_after_backward")

      ctx$last_loss = loss$item()
      predictions[[length(predictions) + 1]] = y_hat$detach()
      indices[[length(indices) + 1]] = as.integer(batch$.index$to(device = "cpu"))
      ctx$optimizer$step()

      call("on_batch_end")
    })

    ctx$last_scores_train = measure_prediction(
      pred_tensor = torch_cat(predictions, dim = 1L),
      measures = ctx$measures_train,
      task = ctx$task_train,
      row_ids = ctx$task_train$row_ids[unlist(indices)],
      prediction_encoder = ctx$prediction_encoder
    )

    call("on_before_valid")
    if (does_validation) {
      ctx$network$eval()
      pred_tensor = torch_network_predict_valid(ctx$network, ctx$loader_valid, call)
      ctx$last_scores_valid = measure_prediction(
        pred_tensor = pred_tensor,
        measures = ctx$measures_valid,
        task = ctx$task_valid,
        row_ids = ctx$task_valid$row_ids,
        prediction_encoder = ctx$prediction_encoder

      )
      ctx$network$train()
    }
    call("on_epoch_end")
  }

  call("on_end")

  # The seed is added later
  list(
    network         = ctx$network,
    loss_fn      = ctx$loss_fn$state_dict(),
    optimizer = ctx$optimizer$state_dict(),
    callbacks       = map(cbs, function(cb) cb$state_dict())
  )
}

has_one_arg = function(network) {
  fargs = formalArgs(network)
  length(fargs) == 1L && !fargs == "..."
}

torch_network_predict_valid = function(network, loader,  callback_receiver = function(step_name) NULL) {
  iter = 1L
  one_arg = has_one_arg(network)
  predictions = vector("list", length = length(loader))
  loop(for (batch in loader) {
    callback_receiver("on_batch_valid_begin")
    predictions[[iter]] = if (one_arg) {
      with_no_grad(network$forward(batch$x[[1L]]))
    } else {
      with_no_grad(invoke(network$forward, .args = batch$x))
    }

    iter = iter + 1L
    callback_receiver("on_batch_valid_end")
  })
  torch_cat(predictions, dim = 1L)
}

torch_network_predict = function(network, loader) {
  iter = 1L

  # If there is one arg, we don't care about the name of the ingress token and we just pass it as
  # an unnamed argument
  # TODO: Maybe we should be stricter, but then we need to ensure that the .getbatch() method of the dataset
  # returns a list where the names of x correspond to the argument names of the network
  one_arg = has_one_arg(network)
  predictions = vector("list", length = length(loader))
  loop(for (batch in loader) {
    predictions[[iter]] = if (one_arg) {
      with_no_grad(network$forward(batch$x[[1L]]))
    } else {
      with_no_grad(invoke(network$forward, .args = batch$x))
    }

    iter = iter + 1L
  })
  torch_cat(predictions, dim = 1L)
}

encode_prediction_default = function(predict_tensor, predict_type, task) {
  # here we assume that the levels of the factors are never reordered!
  # This is important as otherwise all hell breaks loose
  # Currently this check is done in mlr3torch but should at some point be handled in mlr3 / mlr3pipelines

  response = prob = NULL
  if (task$task_type == "classif") {
    if (predict_type == "prob") {
      predict_tensor = with_no_grad(nnf_softmax(predict_tensor, dim = 2L))
    }
    # We still execute the argmax on the device before converting to R
    response = as.integer(with_no_grad(predict_tensor$argmax(dim = 2L))$to(device = "cpu"))

    predict_tensor = predict_tensor$to(device = "cpu")
    if (predict_type == "prob") {
      prob = as.matrix(predict_tensor)
      colnames(prob) = task$class_names
    } else {
      prob = NULL
    }

    class(response) = "factor"
    levels(response) = task$class_names
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
  prediction = as_prediction_data(prediction, task = task, check = TRUE, row_ids = row_ids)
  prediction = as_prediction(prediction, task = task)

  lapply(
    measures,
    function(measure) {
      measure$score(prediction, task = task, train_set = task$row_roles$use)
    }
  )
}
