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

learner_torch_network_output = function(self, private, task, param_vals) {
  # parameter like device "auto" already resolved
  self$network$to(device = param_vals$device)
  self$network$eval()
  data_loader = private$.dataloader_predict(private$.dataset(task, param_vals), param_vals)
  torch_network_predict(self$network, data_loader, device = param_vals$device)
}

learner_torch_predict = function(self, private, super, task, param_vals) {
  network_output = learner_torch_network_output(self, private, task, param_vals)
  encode_network_output(network_output, task, self$predict_type, private$.encode_prediction)
}

encode_network_output = function(network_output, task, predict_type, encoder) {
  # lazy tensors are just simply forwards, irrespective of the provided encoder
  if (predict_type == "lazy_tensor") {
    return(list(lazy_tensor = as_prediction_lazy_tensor(network_output)))
  }
  check_encoded_prediction(encoder(network_output = network_output, task = task), task)
}

as_prediction_lazy_tensor = function(network_output) {
  if (!is.list(network_output)) {
    return(as_lazy_tensor(network_output))
  }
  as.data.table(set_names(lapply(network_output, as_lazy_tensor), head_names(network_output)))
}

head_names = function(network_output) {
  nms = names2(network_output)
  unnamed = is.na(nms) | !nzchar(nms)
  nms[unnamed] = paste0("output_", which(unnamed))
  make.unique(nms)
}

check_encoded_prediction = function(encoded, task) {
  ok = is.list(encoded) && !is.data.frame(encoded) && test_names(names2(encoded), type = "unique")
  if (!ok) {
    stopf("The prediction encoding of task '%s' returned %s, but it has to be a named `list()` with elements such as 'response', 'prob' or 'se'. The encoding is the task's `default_encoder`, unless the learner overwrites the private `.encode_prediction()` method.", task$id, if (is.null(encoded)) "`NULL`" else sprintf("an object of class '%s'", class(encoded)[[1L]])) # nolint
  }
  encoded
}

learner_torch_train = function(self, private, super, task, param_vals) {
  # Here, all param_vals (like seed = "random" or device = "auto") have already been resolved
  is_checkpoint = function(descriptor) identical(descriptor$generator, CallbackSetCheckpoint)
  if (isTRUE(param_vals$resume) && !some(self$callbacks, is_checkpoint)) {
    error_config("Learner '%s' has 'resume' set to TRUE, but no 'checkpoint' callback to take the path from. Either add one via t_clbk(\"checkpoint\") or set 'resume' to a checkpoint folder.", self$id) # nolint
  }
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
  loss_fn = private$.loss_fn(task, param_vals)
  loss_fn$to(device = param_vals$device)

  measures_train = normalize_to_list(param_vals$measures_train)
  measures_valid = normalize_to_list(param_vals$measures_valid)

  if (length(measures_valid) && is.null(self$validate)) {
    stopf("Learner '%s' has measures_valid set, but its validate field is NULL. Configure validation data, e.g. via set_validate(learner, 0.3) or set_validate(learner, \"test\"). If this is happening in an AutoTuner, you may also have forgotten to set epochs = to_tune(upper = <max_epochs>, internal = TRUE).", self$id) # nolint
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
      min_delta = param_vals$min_delta,
      restore_best_weights = param_vals$restore_best_weights
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
  # callbacks are called in the order they were passed, unless they request otherwise via their
  # weight. CallbackSetCheckpoint has weight Inf so that it saves the network and optimizer as the
  # other callbacks left them at the end of the stage.
  # order() breaks ties by the second argument, which keeps the order the callbacks were passed in.
  weights = map_dbl(seq_along(cbs), function(i) {
    assert_number(cbs[[i]]$weight, .var.name = sprintf("weight of callback '%s'", names(cbs)[[i]] %??% i))
  })
  cbs = cbs[order(weights, seq_along(cbs))]
  assert_checkpoint_writes_last(cbs)

  # callbacks such as CallbackSetCheckpoint need access to the other callbacks to save their states,
  # in the order they are called in
  ctx$callbacks = cbs

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


  # if we increment epoch at the end of the loop it has the wrong value
  # during the final two callback stages
  ctx$epoch = 0L
  ctx$global_step = 0L

  resume_path = ctx$learner$param_set$values$resume
  if (!is.null(resume_path)) resume_training(ctx, resume_path)

  call("on_begin")

  ctx$network$train()

  while (!isTRUE(ctx$terminate) && ctx$epoch < ctx$total_epochs) {
    ctx$epoch = ctx$epoch + 1
    call("on_epoch_begin")

    predictions = list()
    indices = list()
    train_iterator = dataloader_make_iter(ctx$loader_train)
    ctx$step = 0L
    eval_train = eval_train_in_epoch(ctx)
    while (ctx$step < length(ctx$loader_train)) {
      ctx$step = ctx$step + 1
      ctx$global_step = ctx$global_step + 1L
      ctx$batch = dataloader_next(train_iterator)
      if (is.null(ctx$batch)) {
        stop("dataloader_next() returned NULL, which means there are no more samples/batches. Typically this occurs when length of sampler/batch_sampler is greater than the number of samples/batches. Please modify .length() method to return the correct number (samples for sampler, batches for batch_sampler), which should be equal to the number of times that .iter() can be called before returning coro::exhausted()")
      }
      ctx$batch$x = lapply(ctx$batch$x, function(x) x$to(device = ctx$device))
      # a task without a target produces batches without a `y`, see `TaskTorch`
      if (!is.null(ctx$batch$y)) {
        ctx$batch$y = ctx$batch$y$to(device = ctx$device)
      }
      ctx$optimizer$zero_grad()

      call("on_batch_begin")

      ctx$y_hats = if (length(ctx$batch$x) == 1L) {
        ctx$network(ctx$batch$x[[1L]])
      } else {
        do.call(ctx$network, ctx$batch$x)
      }
      # A network can return more than one tensor, see the section 'Network Head and Target
      # Encoding' of `LearnerTorch`. `y_hats` is that complete output, which the loss is applied
      # to and which the predictions are encoded from; `y_hat` is its first element, which is
      # offered to callbacks as a convenience.
      ctx$y_hat = if (is.list(ctx$y_hats)) ctx$y_hats[[1L]] else ctx$y_hats

      loss = if (is.null(ctx$batch$y)) {
        ctx$loss_fn(ctx$y_hats)
      } else {
        ctx$loss_fn(ctx$y_hats, ctx$batch$y)
      }

      loss$backward()

      call("on_after_backward")

      ctx$last_loss = loss$item()
      if (eval_train) {
        # The complete output is kept, so `.encode_prediction()` sees the same structure that it
        # sees when predicting. Networks whose extra tensors are training-only, such as auxiliary
        # classifiers, have to reduce them in their `.encode_prediction()` method.
        predictions[[length(predictions) + 1]] = if (is.list(ctx$y_hats)) {
          lapply(ctx$y_hats, function(y_hat) y_hat$detach())
        } else {
          ctx$y_hats$detach()
        }
        indices[[length(indices) + 1]] = as.integer(ctx$batch$.index$to(device = "cpu"))
      }
      ctx$optimizer$step()

      call("on_batch_end")
    }

    ctx$last_scores_train = if (eval_train) {
      measure_prediction(
        network_output = cat_predictions(predictions),
        measures = ctx$measures_train,
        task = ctx$task_train,
        row_ids = ctx$task_train$row_ids[unlist(indices)],
        prediction_encoder = ctx$prediction_encoder,
        predict_type = ctx$learner$predict_type
      )
    }

    call("on_before_valid")
    if (eval_valid_in_epoch(ctx)) {
      ctx$network$eval()
      network_output = torch_network_predict_valid(ctx, call)
      ctx$last_scores_valid = measure_prediction(
        network_output = network_output,
        measures = ctx$measures_valid,
        task = ctx$task_valid,
        row_ids = ctx$task_valid$row_ids,
        prediction_encoder = ctx$prediction_encoder,
        train_set = ctx$task_train$row_roles$use,
        predict_type = ctx$learner$predict_type
      )
      ctx$network$train()
      call("on_valid_end")
    } else {
      ctx$last_scores_valid = NULL
    }
    call("on_epoch_end")
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

resume_training = function(ctx, resume_path) {
  path = if (isTRUE(resume_path)) checkpoint_callback_path(ctx$callbacks) else resume_path

  checkpoint = latest_checkpoint(path)
  if (is.null(checkpoint)) {
    if (!can_checkpoint_into(path)) {
      stopf("No checkpoint to resume from in '%s': it does not exist or holds no checkpoint written by t_clbk(\"checkpoint\"). Point 'resume' at a folder that one wrote, or unset it to train from scratch.", path) # nolint
    }
    lg$info("No checkpoint found in '%s', starting training from scratch.", path)
    return(invisible(NULL))
  }
  epochs_trained = checkpoint$epoch
  if (epochs_trained > ctx$total_epochs) {
    stopf("The checkpoint in '%s' was already trained for %i epochs, but 'epochs' is %i. Note that 'epochs' is the total number of epochs, including those of the checkpoint, so it cannot be less than %i.", # nolint
      path, epochs_trained, ctx$total_epochs, epochs_trained)
  }
  state = read_checkpoint_state(checkpoint$state)
  assert_resumable_task(ctx, path)

  if (epochs_trained == ctx$total_epochs) {
    lg$info("The checkpoint in '%s' is at epoch %i, which is 'epochs', so this run trains nothing and returns the model of the checkpoint.", path, epochs_trained) # nolint
  } else {
    lg$info("Resuming training from the checkpoint in '%s', which is at epoch %i.", path, epochs_trained)
  }

  ctx$network$load_state_dict(torch_load(checkpoint$network))
  ctx$optimizer$load_state_dict(torch_load(checkpoint$optimizer))
  load_callback_states(ctx$callbacks, state)

  ctx$epoch = epochs_trained
  ctx$global_step = state$global_step
  ctx$last_scores_valid = state$valid_scores

  invisible(NULL)
}

assert_checkpoint_writes_last = function(cbs) {
  checkpoints = which(map_lgl(cbs, function(cb) inherits(cb, "CallbackSetCheckpoint")))
  if (!length(checkpoints)) {
    return(invisible(NULL))
  }
  stale = keep(cbs[-seq_len(max(checkpoints))], function(cb) {
    !identical(body(cb$state_dict), body(CallbackSet$public_methods$state_dict)) &&
      any(c("on_epoch_end", "on_end") %in% cb$stages)
  })
  if (length(stale)) {
    stopf("Callback(s) %s update their state at the end of an epoch but run after the 'checkpoint' callback, which would store the state they had one epoch earlier. Give them a lower $weight than the checkpoint callback, see the section 'Ordering' of CallbackSet.", # nolint
      paste0("'", names(stale), "'", collapse = ", "))
  }
  invisible(NULL)
}

checkpoint_callback_path = function(cbs) {
  cbs = keep(cbs, function(cb) inherits(cb, "CallbackSetCheckpoint"))
  if (!length(cbs)) {
    error_config("'path' is TRUE, but there is no 'checkpoint' callback to take the path from.")
  }
  cbs[[1L]]$path
}

assert_resumable_task = function(ctx, path) {
  state = readRDS(file.path(path, "run.rds"))
  if (!identical(state$task_id, ctx$task_train$id)) {
    stopf("The checkpoint in '%s' was written for task '%s', but this run trains on '%s'. Resume with the task the checkpoint was written for.", # nolint
      path, state$task_id, ctx$task_train$id)
  }
  valid_ids = if (!is.null(ctx$task_valid)) ctx$task_valid$row_ids
  if (xor(is.null(state$valid_row_ids), is.null(valid_ids))) {
    stopf("The checkpoint in '%s' was written %s an internal validation split, but this run has %s one. Resume with the validation configuration the checkpoint was written with.", # nolint
      path, if (is.null(state$valid_row_ids)) "without" else "with",
      if (is.null(valid_ids)) "no" else "such")
  }
  if (!test_permutation(state$valid_row_ids, valid_ids)) {
    stopf("The checkpoint in '%s' was written for a different internal validation split: %i of its %i validation rows are also validation rows of this run. Note that `validate = <ratio>` draws a new split from R's random number generator in every run, which the 'seed' parameter does not govern. Use validate = \"predefined\" with a fixed internal validation task, or seed R's generator identically before each run.", # nolint
      path, length(intersect(state$valid_row_ids, valid_ids)), length(state$valid_row_ids))
  }
  invisible(NULL)
}

load_callback_states = function(cbs, state) {
  states = state$callbacks
  if (!length(states)) return(invisible(NULL))
  unknown = setdiff(names(states), names(cbs))
  if (length(unknown)) {
    warningf("The checkpoint contains states for callback(s) %s, which are not part of this training run. They are ignored.", # nolint
      paste0("'", unknown, "'", collapse = ", "))
  }
  shared = intersect(names(states), names(cbs))
  mismatch = keep(shared, function(id) {
    id %in% names(state$callback_classes) && !identical(class(cbs[[id]])[[1L]], state$callback_classes[[id]])
  })
  if (length(mismatch)) {
    stopf("The callbacks of this run are not the ones the checkpoint stored a state for: %s. States are matched by id, so restoring would feed the state of one callback into another. Resume with the callbacks the checkpoint was written with, or give the new ones ids of their own.", # nolint
      paste0(map_chr(mismatch, function(id) {
        sprintf("'%s' was a <%s> and is a <%s>", id, state$callback_classes[[id]], class(cbs[[id]])[[1L]])
      }), collapse = ", "))
  }
  iwalk(states[shared], function(state, id) cbs[[id]]$load_state_dict(state))
  invisible(NULL)
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
  length(fargs) == 1L && fargs != "..."
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
  cat_predictions(predictions)
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
  cat_predictions(predictions)
}

zero_row_network_output = function(learner, task) {
  if (is.null(learner$model)) {
    stopf("Learner '%s' cannot build the empty prediction for task '%s' because it has no model: what a network returns for zero rows is known to the network alone, and the task cannot be asked instead.", learner$id, task$id) # nolint
  }
  # probe with a single batch and set the resulting batch dims to 0
  # we can't rely on output_dim_for, as this is just the default and learners can overwrite it.
  tryCatch({
    param_vals = learner$param_set$get_values(tags = "predict")
    param_vals$device = auto_device(param_vals$device)
    # `mlr3` takes this path before it unmarshals, so the model may still be marshaled here
    model = if (isTRUE(learner$marshaled)) {
      unmarshal_model(learner$model, inplace = FALSE)
    } else {
      learner$model
    }
    network = model$network
    network$to(device = param_vals$device)
    network$eval()
    # `mlr3` filtered the task to zero rows, but its backend still holds them
    rows = task$backend$rownames
    if (!length(rows)) {
      stopf("the task has no rows at all, so no batch can be built from it")
    }
    one_row = task$clone(deep = TRUE)
    one_row$row_roles$use = rows[1L]
    batch = get_private(learner)$.dataset(one_row, param_vals)$.getbatch(1L)
    x = lapply(batch$x, function(tensor) tensor$to(device = param_vals$device))
    output = if (has_one_arg(network)) {
      with_no_grad(network$forward(x[[1L]]))
    } else {
      with_no_grad(invoke(network$forward, .args = x))
    }
    # only the structure is wanted, so the one row is cut away again
    if (is.list(output)) lapply(output, drop_rows) else drop_rows(output)
  }, error = function(e) {
    stopf("Learner '%s' cannot build the empty prediction for task '%s': running its network on a batch of one row failed with: %s", learner$id, task$id, conditionMessage(e)) # nolint
  })
}

drop_rows = function(tensor) {
  tensor$narrow(1L, 1L, 0L)
}

# The network's evaluation-mode output is what `encode_prediction()` receives: either a single
# tensor or, for a network with more than one head, a `list()` of them. The batches are concatenated
# head-wise, so the encoder sees the same structure it would see for a single batch.
cat_predictions = function(predictions) {
  first = predictions[[1L]]
  if (!is.list(first)) {
    return(torch_cat(predictions, dim = 1L))
  }
  if (!length(first)) {
    stopf("The network returned an empty list, it must return a tensor or a non-empty list of tensors.")
  }
  ok = map_lgl(predictions, function(batch) is.list(batch) && length(batch) == length(first))
  if (!all(ok)) {
    stopf("The network returned %i tensors for the first batch but something else for batch %i, it must return the same number of tensors for every batch.", length(first), which(!ok)[[1L]]) # nolint
  }
  heads = lapply(seq_along(first), function(i) {
    torch_cat(lapply(predictions, function(batch) batch[[i]]), dim = 1L)
  })
  names(heads) = names(first)
  heads
}

# The built-in task types have a single network head, so a one-element list is unwrapped and
# anything longer is an output that these encodings cannot interpret.
assert_single_head = function(network_output, task) {
  if (!is.list(network_output)) {
    return(network_output)
  }
  if (length(network_output) != 1L) {
    stopf("The network returned %i tensors, but the prediction encoding for task type '%s' expects a single one. Overwrite the learner's private `.encode_prediction()` method to combine them.", length(network_output), task$task_type) # nolint
  }
  network_output[[1L]]
}

#' @title Encode the Network Output as a Prediction
#'
#' @description
#' Converts the raw output of a network into a `list()` that can be passed to
#' [`mlr3::as_prediction_data()`], which is what the private `.encode_prediction()` method of a
#' [`LearnerTorch`] has to return.
#'
#' This is the default implementation that is used by [`LearnerTorch`] and
#' [`LearnerTorchModel`][mlr_learners_torch_model], i.e. by all learners that don't overwrite
#' `.encode_prediction()`.
#' When adding support for a custom task type, implement a method for the corresponding
#' [`Task`][mlr3::Task] class, which makes the generic torch learners work for that task type.
#'
#' For the network output that is expected for the built-in task types, see section
#' *Network Head and Target Encoding* of [`LearnerTorch`].
#'
#' @param task ([`Task`][mlr3::Task])\cr
#'   The task to predict on.
#' @param network_output ([`torch_tensor`][torch::torch_tensor] or `list()` of them)\cr
#'   The raw output of the network in evaluation mode.
#'   A network with more than one head -- e.g. one predicting a mean and a standard deviation --
#'   returns a `list()` of tensors, which is passed on unchanged.
#'   The encodings of the built-in task types expect a single tensor.
#' @param predict_type (`character(1)`)\cr
#'   The predict type of the learner, e.g. `"response"` or `"prob"`.
#' @param ... (any)\cr
#'   Additional arguments. Not used yet.
#' @return named `list()`
#' @export
encode_prediction = function(task, network_output, predict_type, ...) {
  UseMethod("encode_prediction")
}

#' @export
encode_prediction.default = function(task, network_output, predict_type, ...) { # nolint
  stopf("No prediction encoding available for task type '%s', implement an `encode_prediction()` method for class '%s' or overwrite the learner's private `.encode_prediction()` method.", task$task_type, class(task)[[1L]]) # nolint
}

#' @export
encode_prediction.TaskClassif = function(task, network_output, predict_type, ...) { # nolint
  network_output = assert_single_head(network_output, task)
  # here we assume that the levels of the factors are never reordered!
  # This is important as otherwise all hell breaks loose
  # Currently this check is done in mlr3torch but should at some point be handled in mlr3 / mlr3pipelines

  response = prob = NULL
  if ("multiclass" %in% task$properties) {
    if (predict_type == "prob") {
      network_output = with_no_grad(nnf_softmax(network_output, dim = 2L))
    }
    # We still execute the argmax on the device before converting to R
    response = as.integer(with_no_grad(network_output$argmax(dim = 2L))$to(device = "cpu"))

    network_output = network_output$to(device = "cpu")
    prob = if (predict_type == "prob") {
      prob = as.matrix(network_output)
      colnames(prob) = task$class_names
      prob
    }

    class(response) = "factor"
    levels(response) = task$class_names
    return(list(response = response, prob = prob))
  } else {
    # binary:
    # (first factor level is positive class)
    response = as.integer(with_no_grad(network_output < 0)$to(device = "cpu") + 1)
    class(response) = "factor"
    levels(response) = task$class_names

    prob = if (predict_type == "prob") {
      # convert score to prob
      network_output = with_no_grad(nnf_sigmoid(network_output))
      prob = as.numeric(network_output)
      prob = as.matrix(data.frame(prob, 1 - prob))
      colnames(prob) = task$class_names
      prob
    }

    return(list(response = response, prob = prob))
  }
}

#' @export
encode_prediction.TaskRegr = function(task, network_output, predict_type, ...) { # nolint
  if (predict_type != "response") {
    stopf("Invalid predict_type for task_type 'regr'.")
  }
  list(response = as.numeric(assert_single_head(network_output, task)))
}


measure_prediction = function(network_output, measures, task, row_ids, prediction_encoder,
  train_set = task$row_roles$use, predict_type) {
  if (!length(measures)) {
    return(structure(list(), names = character(0)))
  }

  prediction = encode_network_output(network_output, task, predict_type, prediction_encoder)
  # tagged at the point of use rather than in `.encode_prediction()`, which a learner may overwrite
  # -- an overwritten one would silently produce a prediction without a truth
  class(prediction) = c("prediction_torch", "list")
  prediction = as_prediction_data(prediction, task = task, row_ids = row_ids, check = FALSE)
  prediction = as_prediction(prediction, task = task, check = FALSE)

  lapply(
    measures,
    function(measure) {
      tryCatch(
        measure$score(prediction, task = task, train_set = train_set),
        error = function(e) {
          warningf("Measure '%s' could not be computed and is reported as NaN: %s", measure$id, conditionMessage(e)) # nolint
          NaN
        }
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

#' @export
print.learner_torch_model = function(x, ...) {
  n_params = if (!is.null(x$network)) sum(map_dbl(x$network$parameters, function(p) prod(dim(p))))

  catn(sprintf("<learner_torch_model> trained for %s epoch%s",
    x$epochs %??% "?", if (isTRUE(x$epochs == 1)) "" else "s"))
  catn(str_indent("* Network: ", if (is.null(x$network)) {
    "- (the model is marshaled)"
  } else {
    sprintf("<%s> with %s parameters", class(x$network)[[1L]], format(n_params, big.mark = ","))
  }))
  catn(str_indent("* Callbacks: ", if (length(x$callbacks)) paste0(names(x$callbacks), collapse = ", ") else "-"))
  if (length(x$internal_valid_scores)) {
    scores = sprintf("%s = %s", names(x$internal_valid_scores),
      format(unlist(x$internal_valid_scores), digits = 4L))
    catn(str_indent("* Validation scores: ", paste0(scores, collapse = ", ")))
  }
  catn(str_indent("* Fields: ", paste0(names(x), collapse = ", ")))
  invisible(x)
}
