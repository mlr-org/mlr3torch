LearnerTorch = R6Class("DeepLearner",
  inherit = Learner,
  public = list(
    initialize = function(task_type, predict_types, param_set, properties) {
      super$initialize(
        id = sprintf("%s.torch", task_type),
        feature_types = c("numeric", "integer", "factor", "ordered"),
        predict_type = predict_types,
        packages = "torch",
        param_set = param_set,
        properties = properties,
        man = "mlr3torch::mlr_learners_deep_learner",
        task_type = task_type
      )
    }
  ),
  private = list(
    .train = function(task) {
      if (!length(self$state)) {
        private$.build(task)
      }
      if (self$param_set$values$n_epochs > 0L) {
        assert(!is.null(self$state$optimizer))
        assert(!is.null(self$state$criterion))
        train_model(
          model = self$state$model,
          task = task,
          optimizer = self$state$optimizer,
          criterion = self$state$criterion,
          n_epochs = self$param_set$values$n_epochs,
          batch_size = self$param_set$values$batch_size,
          device = self$param_set$values$device
        )
      }
      return(self$state$model)
    },

    .build = function(task) {
      pars = self$param_set$get_values(tag = "train")
      reduction = reduce_architecture(pars[["architecture"]], task)
      model = reduction[["model"]]
      self$state = list(
        model = model,
        optimizer = mlr3misc::invoke(pars[["optimizer"]], .args = pars[["optimizer_args"]],
          params = model$parameters
        ),
        criterion = mlr3misc::invoke(pars[["criterion"]], .args = pars[["criterion_args"]])
      )
    },

    .predict = function(task) {
      assert(task$task_type == "regr")
      pars = self$param_set$get_values(tags = "predict")
      response = predict_from_model(self$state$model, task, pars$device, pars$batch_size)
      response
      # predictions = predict_from_model(model, task)
      # list(response = predictions)
      # newdata = task$data(cols = task$feature_names)
      # response = invoke(predict, self$model, newdata = newdata, .opts = allow_partial_matching)
      # list(response = unname(response))
    }
  )
)

predict_from_model = function(model, task, device, batch_size) {
  task_type = task$task_type
  y_hats = list()
  data_loader = make_dataloader(task, batch_size, device)

  coro::loop(for (batch in data_loader) {
    xs = batch[startsWith(names(batch), "x")]
    if (length(xs) == 1L) {
      xs = xs[[1]]
    }
    y_hat = with_no_grad(model$forward(xs))[NULL]
    y_hats = append(y_hats, y_hat)
  })

  y_hats = torch_cat(y_hats)
  # y_hats = switch(task_type,
  #   classif = factor()
  # )
  # y_hats = as.numeric(y_hats)
  # if
  # return(y_hats)
  return(y_hats)
}

train_model = function(model, task, optimizer, criterion, n_epochs, batch_size, device) {
  # TODO: handle device and training / evaluation state of nn_module (batchnorm etc.)
  data_loader = make_dataloader(task, batch_size, device)
  # iterator = data_loader$.iter()
  # batch = iterator$.next()
  # optimizer$zero_grad()
  # xs = batch[startsWith(names(batch), "x")]
  # if (length(xs) == 1L) {
  #   xs = xs[[1]]
  # }


  for (epoch in seq_len(n_epochs)) {
    losses = list()
    coro::loop(for (batch in data_loader) {
      optimizer$zero_grad()
      xs = batch[startsWith(names(batch), "x")]
      if (length(xs) == 1L) {
        xs = xs[[1]]
      }
      y_hat = model$forward(xs)
      if (inherits(criterion, "nn_crossentropy_loss")) {
        y_true = batch$y[,1]
      } else {
        y_true = batch$y
      }
      loss = criterion(y_hat, y_true)
      loss$backward()
      optimizer$step()
      losses = c(losses, loss$item())
    })
  }

}
