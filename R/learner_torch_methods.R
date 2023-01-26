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

learner_torch_train = function(self, task) {
  private = self$.__enclos_env__$private
  super = self$.__enclos_env__$super
  param_vals = self$param_set$get_values(tags = "train")
  learner_torch_train_worker(self, private, super, task, param_vals, FALSE)
}


# learner_torch_continue = function(self, task) {
#   private = self$.__enclos_env__$private
#   super = self$.__enclos_env__$super
#   param_vals = self$param_set$get_values()
#   param_vals = set_defaults()
#   param_vals$epochs = assert_int(param_vals$epochs - self$state$param_vals$epochs, lower = 1)
#   learner_torch_train_worker(self, private, super, task, param_vals, TRUE)
# }


learner_torch_train_worker = function(self, private, super, task, param_vals, continue = FALSE) {
  torch_set_num_threads(param_vals$num_threads %??% 1L)

  loader_train = private$.dataloader(task, param_vals)

  task_valid = task$clone()$filter(integer(0))
  task_valid$set_row_roles(task$row_roles$test, "use")
  loader_valid = if (task_valid$nrow) private$.dataloader(task_valid, insert_named(param_vals, list(shuffle = FALSE)))

  if (continue) {
    network = self$model$network
    optimizer = self$model$optimizer
    loss_fn = self$model$loss_fn
    callbacks = self$model$callbacks
  } else {
    network = private$.network(task, param_vals)$to(device = param_vals$device)
    optimizer = private$.optimizer$get_optimizer(network$parameters)
    loss_fn = private$.loss$get_loss()
    callbacks = normalize_to_list(c(list(CallbackTorchHistory$new()), param_vals$callbacks))
  }

  ctx = ContextTorch$new(
    learner = self,
    task_train = task,
    task_valid = if (task_valid$nrow) task_valid,
    loader_train = loader_train,
    loader_valid = loader_valid,
    measures_train = normalize_to_list(param_vals$measures_train),
    measures_valid = if (task_valid$nrow) normalize_to_list(param_vals$measures_valid) else list(),
    network = network,
    optimizer = optimizer,
    loss_fn = loss_fn,
    total_epochs = param_vals$epochs
  )

  train_loop(ctx, callbacks)
}


train_loop = function(ctx, cbs) {
  call = function(step_name) {
    lapply(cbs, function(x) if (!is.null(x[[step_name]])) x[[step_name]](ctx))
  }

  ## we do this so if the learner should crash the intermediate progress is saved somewhere
  ctx$learner$state$model = list(
    network = ctx$network,
    optimizer = ctx$optimizer,
    loss_fn = ctx$loss_fn,
    callbacks = cbs
  )

  # note that task_valid may be present (callbacks could do their own validation)
  does_validation = length(ctx$measures_valid)

  walk(cbs, function(cb) cb$state = NULL)

  on.exit({
    # in case a callback wants to finalize things
    call("on_end")
  }, add = TRUE)


  call("on_begin")

  ctx$network$train()

  ctx$epoch = 0
  while (ctx$epoch < ctx$total_epochs) {
    ctx$epoch = ctx$epoch + 1
    call("on_epoch_begin")

    predictions = list()
    indices = list()
    ctx$batch = 0
    coro::loop(for (batch in ctx$loader_train) {
      ctx$batch = ctx$batch + 1

      ctx$optimizer$zero_grad()

      call("on_batch_begin")

      if (length(batch$x) == 1L) {
        # No need to match the names of the forward function with the return of the dataloader as there is no
        # ambiguity
        y_hat = ctx$network(batch$x[[1L]])
      } else {
        y_hat = do.call(ctx$network, batch$x)
      }

      loss = ctx$loss_fn(y_hat, batch$y)

      loss$backward()

      call("on_after_backward")

      ctx$last_loss = loss$item()
      predictions[[length(predictions) + 1]] = y_hat$detach()
      indices[[length(indices) + 1]] = as.numeric(batch$.index)
      ctx$optimizer$step()

      call("on_batch_end")
    })

    ctx$last_scores_train = measure_prediction(torch_cat(predictions, dim = 1L), ctx$measures_train, ctx$task_train,
      ctx$task_train$row_ids[unlist(indices)])

    call("on_before_validation")
    if (does_validation) {
      ctx$network$eval()
      pred_tensor = torch_network_predict(ctx$network, ctx$loader_valid, call)
      ctx$last_scores_valid = measure_prediction(pred_tensor, ctx$measures_valid, ctx$task_valid,
        ctx$task_valid$row_ids)
      ctx$network$train()
    }
    call("on_epoch_end")
  }

  list(
    network = ctx$network,
    optimizer = ctx$optimizer,
    loss_fn = ctx$loss_fn,
    callbacks = cbs
  )
}

torch_network_predict = function(network, loader, callback_receiver = function(step_name) NULL) {
  iter = 1L
  predictions = list()
  one_arg = length(formalArgs(network)) == 1L
  loop(for (batch in loader) {
    callback_receiver("on_batch_valid_begin")
    if (one_arg) {
      predictions[[iter]] = with_no_grad(network(batch$x[[1L]]))
    } else {
      predictions[[iter]] = with_no_grad(do.call(network, batch$x))
    }
    iter = iter + 1L
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
      levels(response) = task$levels(task$target_names)[[1L]]
    } else if (predict_type == "prob") {
      predict_tensor = nnf_softmax(predict_tensor, dim = 2L)
      prob = as.matrix(predict_tensor)
      colnames(prob) = task$class_names
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


measure_prediction = function(pred_tensor, measures, task, row_ids) {
  if (!length(measures)) {
    return(structure(list(), names = character(0)))
  }

  prediction = encode_prediction(pred_tensor, "prob", task)
  prediction = as_prediction_data(prediction, task = task, check = TRUE, row_ids = row_ids)
  prediction = as_prediction(prediction, task = task)

  lapply(
    measures,
    function(measure) {
      measure$score(prediction, task = task, train_set = task$row_roles$use)
    }
  )
}

# Here are the standard methods that are shared between all the TorchLearners
learner_torch_predict = function(self, task) {
  private = self$.__enclos_env__$private
  model = self$state$model
  network = model$network
  network$eval()

  param_vals = self$param_set$get_values(tags = "predict")

  param_vals$shuffle = FALSE

  data_loader = private$.dataloader(task, param_vals)

  prediction = torch_network_predict(network, data_loader)

  encode_prediction(prediction, self$predict_type, task)
}

learner_torch_network = function(self, task, rhs) {
  assert_ro_binding(rhs)
  if (is.null(self$state)) {
    stopf("Cannot access network before training.")
  }
  self$state$model$network
}

learner_torch_param_set = function(self, rhs) {
  private = self$.__enclos_env__$private
  if (is.null(private$.param_set)) {
    private$.param_set = ParamSetCollection$new(
      list(private$.param_set_base, private$.optimizer$param_set, private$.loss$param_set))
  }
  private$.param_set
}

learner_torch_hist_valid = function(self, rhs) {
  assert_ro_binding(rhs)
  if (is.null(self$state)) {
    stopf("Cannot access validation history before training.")
  }
  self$model$callbacks$history$state$hist_valid
}

learner_torch_hist_train = function(self, rhs) {
  assert_ro_binding(rhs)
  if (is.null(self$state)) {
    stopf("Cannot access training history before training.")
  }
  self$model$callbacks$history$state$hist_train
}
