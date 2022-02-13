LearnerTorch = R6Class("DeepLearner",
  inherit = Learner,
  public = list(
    initialize = function(task_type, predict_types, param_set, properties) {
      super$initialize(
        id = sprintf("%s.torch", task_type),
        feature_types = c("numeric", "integer", "character", "ordered", "factor"),
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
      train_model(
        model = self$state$model,
        task = task,
        optimizer = self$state$optimizer,
        criterion = self$state$criterion,
        n_epochs = self$param_set$values$n_epochs,
        batch_size = self$param_set$values$batch_size,
        device = self$param_set$values$device
      )
      return(self$state$model)
    },

    .build = function(task) {
      pars = self$param_set$get_values(tag = "train")
      model = reduce_architecture(pars[["architecture"]], task)[["model"]]
      self$state = list(
        model = model,
        optimizer = mlr3misc::invoke(pars[["optimizer"]], .args = pars[["optimizer_args"]],
          params = model$parameters
        ),
        criterion = mlr3misc::invoke(pars[["criterion"]], .args = pars[["criterion_args"]])
      )
    },

    .predict = function(task) {
      stop("Not implemented yet!")
      newdata = task$data(cols = task$feature_names)
      response = invoke(predict, self$model, newdata = newdata, .opts = allow_partial_matching)
      list(response = unname(response))
    }
  )
)

train_model = function(model, task, optimizer, criterion, n_epochs, batch_size, device) {
  # TODO: handle device and training / evaluation state of nn_module (batchnorm etc.)
  browser()
  data_loader = make_dataloader(task, batch_size, device)

  for (epoch in seq_len(n_epochs)) {
    losses = list()
    coro::loop(for (batch in data_loader) {
      optimizer$zero_grad()
      xs = batch$x
      output = model$forward(batch$x$to(device = device))
      loss = criterion(y_hat, batch$y$to(device = device))
      optimizer$step()
      losses = c(losses, loss$item())
    })
  }

}
