# Here are the standard methods that are shared between all the TorchLearners
learner_torch_predict = function(self, task) {
  model = self$state$model
  reset_train = model$network$training
  on.exit(if (reset_train) model$network$train(), add = TRUE)
  model$network$eval()

  network = model$network

  pars = self$param_set$get_values(tags = "predict")
  device = pars$device
  batch_size = pars$batch_size

  data_loader = as_dataloader(task, device = device, batch_size = batch_size, drop_last = FALSE)
  npred = length(data_loader$dataset) # length of dataloader are the batches
  responses = integer(npred)
  i = 0L
  coro::loop(for (batch in data_loader) {
    p = with_no_grad(
      network$forward(batch$x)
    )
    p = as.integer(p$argmax(dim = 2L)$to(device = "cpu"))
    # TODO: differentiate between different predict types
    responses[(i * batch_size + 1L):min(((i + 1L) * batch_size), npred)] = p
    i = i + 1L
  })

  # TODO: Check that nothing goes wrong here
  class(responses) = "factor"
  levels(responses) = task$levels(cols = task$target_names)[[1L]]
  list(response = responses)
}
