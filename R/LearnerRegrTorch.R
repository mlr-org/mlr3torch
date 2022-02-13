LearnerTorchRegr = R6Class("LearnerTorchRegr",
  inherit = LearnerRegr,
  public = list(
    initialize = function() {
      super$initialize(
        id = "torch.classif",
        feature_types = c("numeric", "integer", "character", "logical", "factor"),
        predict_type = "response",
        packages = "torch",
        param_set = dl_paramset(),
        properties = c(), # TODO: weights, importance
        man = "mlr3torch::mlr_learners_deep_learner" # TODO: fix manual
      )
    }
  ),
  private = list(
    .train = function(task) {
      browser()
      if (!length(self$state)) {
        private$.build(task)
      }
      train_model(
        network = self$state$model,
        task = task,
        optimizer = self$state$optimizer,
        criterion = self$state$criterion,
        n_epochs = self$param_set$values$n_epochs,
        batch_size = self$param_set$values$batch_size,
        device = self$param_set$values$device
      )
    },

    .build = function(task) {
      browser()
      pars = self$param_set$get_values(tag = "train")
      self$state = list(
        model = reduce_architecture(pars[["architecture"]], task),
        optimizer = mlr3misc::invoke(pars[["optimizer"]], .args = pars[["optimizer_args"]]),
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
