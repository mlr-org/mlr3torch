# Here are the standard methods that are shared between all the TorchLearners
learner_torch_predict = function(self, task) {
  model = self$state$model
  network = model$network
  network$eval()

  p = self$param_set$get_values(tags = "predict")

  data_loader = as_dataloader(task, device = p$device, batch_size = p$batch_size, drop_last = FALSE)

  predictions = vector("list", length(dataloader))
  i = 1L
  coro::loop(for (batch in data_loader) {
    predictions[[i]] = with_no_grad(network$forward(batch$x))
    i = i + 1L
  })

  prediction = torch_cat(predictions, dim = 1L)

  encode_prediction(prediction, self$predict_type, task)
}

