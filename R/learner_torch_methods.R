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

  seed = param_vals$seed %??% sample.int(10000000L, 1L)
  torch_manual_seed(seed)

  withr::with_seed(seed = seed, code = {
    learner_torch_train_worker(self, private, super, task, param_vals, FALSE, seed = seed) })
}


# learner_torch_continue = function(self, task) {
#   private = self$.__enclos_env__$private
#   super = self$.__enclos_env__$super
#   param_vals = self$param_set$get_values()
#   param_vals = set_defaults()
#   param_vals$epochs = assert_int(param_vals$epochs - self$state$param_vals$epochs, lower = 1)
#   learner_torch_train_worker(self, private, super, task, param_vals, TRUE)
# }

learner_torch_initialize = function(
  self,
  private,
  super,
  task_type,
  id,
  optimizer,
  loss,
  param_set,
  properties,
  packages,
  predict_types,
  feature_types,
  man,
  label,
  callbacks
  ) {
  private$.optimizer = as_torch_optimizer(optimizer, clone = TRUE)
  private$.optimizer$param_set$set_id = "opt"

  private$.loss = as_torch_loss(loss, clone = TRUE)
  private$.loss$param_set$set_id = "loss"

  callbacks = as_torch_callbacks(callbacks, clone = TRUE)
  callback_ids = ids(callbacks)
  assert_names(callback_ids, type = "unique")

  private$.callbacks = set_names(callbacks, ids(callbacks))
  walk(private$.callbacks, function(cb) {
    cb$param_set$set_id = paste0("cb.", cb$id)
  })

  # TODO: Here we should tag all the parameters of the callbacks and optimizer and loss with `"train"` (?)

  packages = unique(c(
    packages,
    unlist(map(private$.callbacks, "packages")),
    private$.loss$packages,
    private$.optimizer$packages
  ))

  assert_subset(properties, mlr_reflections$learner_properties[[task_type]])
  assert_subset(predict_types, names(mlr_reflections$learner_predict_types[[task_type]]))
  if (any(grepl("^(loss\\.|opt\\.|cb\\.)", param_set$ids()))) {
    stopf("Prefixes 'loss.', 'opt.', and 'cb.' are reserved for dynamically constructed parameters.")
  }
  packages = assert_character(packages, any.missing = FALSE, min.chars = 1L)
  packages = union(c("mlr3", "mlr3torch"), packages)

  paramset_torch = paramset_torchlearner()
  if (param_set$length > 0) {
    private$.param_set_base = ParamSetCollection$new(list(param_set, paramset_torch))
  } else {
    private$.param_set_base = paramset_torch
  }

  super$initialize(
    id = id,
    packages = packages,
    param_set = self$param_set,
    predict_types = predict_types,
    properties = properties,
    data_formats = "data.table",
    label = label,
    feature_types = feature_types,
    man = man
  )

}


learner_torch_train_worker = function(self, private, super, task, param_vals, continue = FALSE, seed = seed) {
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
    optimizer = private$.optimizer$generate(network$parameters)
    loss_fn = private$.loss$generate()

    callbacks = c(lapply(private$.callbacks, function(cb) cb$generate()))
    callbacks = set_names(callbacks, ids(private$.callbacks))
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

  train_loop(ctx, callbacks, seed = seed)
}


train_loop = function(ctx, cbs, seed) {
  call = function(step_name) {
    lapply(cbs, function(x) {
      if (exists(step_name, x, inherits = FALSE)) {
        x[[step_name]](ctx)
      }
    })
  }

  ## we do this so if the learner should crash the intermediate progress is saved somewhere
  ctx$learner$state$model = list(
    network = ctx$network,
    optimizer = ctx$optimizer,
    loss_fn = ctx$loss_fn,
    callbacks = cbs,
    seed = seed
  )

  # note that task_valid may be present (callbacks could do their own validation)
  does_validation = length(ctx$measures_valid)

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
        # With one argument there is no ambiguity and we can be less strict
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

    ctx$last_scores_train = measure_prediction(
      pred_tensor = torch_cat(predictions, dim = 1L),
      measures = ctx$measures_train,
      task = ctx$task_train,
      row_ids = ctx$task_train$row_ids[unlist(indices)]
    )

    call("on_before_valid")
    if (does_validation) {
      ctx$network$eval()
      pred_tensor = torch_network_predict_valid(ctx$network, ctx$loader_valid, call)
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
    callbacks = cbs,
    seed = seed
  )
}

torch_network_predict_valid = function(network, loader,  callback_receiver = function(step_name) NULL) {
  iter = 1L
  predictions = vector("list", length = length(loader))
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

torch_network_predict = function(network, loader) {
  iter = 1L
  predictions = vector("list", length = length(loader))
  one_arg = length(formalArgs(network$forward)) == 1L
  loop(for (batch in loader) {
    if (one_arg) {
      predictions[[iter]] = with_no_grad(network$forward(batch$x[[1L]]))
    } else {
      predictions[[iter]] = with_no_grad(invoke(network$forward, .args = batch$x))
    }
    iter = iter + 1L
  })
  torch_cat(predictions, dim = 1L)

}

encode_prediction = function(predict_tensor, predict_type, task) {

  response = prob = NULL
  if (task$task_type == "classif") {
    if (predict_type == "prob") {
      predict_tensor = nnf_softmax(predict_tensor, dim = 2L)
    }
    # We still execute the argmax on the device before converting to R
    response = as.integer(predict_tensor$argmax(dim = 2L))

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
  param_vals = self$param_set$get_values(tags = "predict")
  param_vals$shuffle = FALSE

  seed = self$model$seed
  torch_manual_seed(seed)
  state = withr::with_seed(seed = seed, code = {
    network$eval()
    data_loader = private$.dataloader(task, param_vals)
    prediction = torch_network_predict(network, data_loader)
    encode_prediction(prediction, self$predict_type, task)
  })
  # TODO: Set torch seed back as well
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
    private$.param_set = ParamSetCollection$new(c(
      list(private$.param_set_base, private$.optimizer$param_set, private$.loss$param_set),
      map(private$.callbacks, "param_set"))
    )
  }
  private$.param_set
}

learner_torch_history = function(self, rhs) {
  assert_ro_binding(rhs)
  if (is.null(self$state)) {
    stopf("Cannot access history before training.")
  }
  if (is.null(self$model$callbacks$history)) {
    warningf("No history found. Did you specify t_clbk(\"history\") during construction?")
    return(NULL)
  }

  self$model$callbacks$history
}
