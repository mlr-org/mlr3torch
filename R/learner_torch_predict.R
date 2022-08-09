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

encode_prediction = function(prediction, predict_type, task) {
  response = prob = NULL
  if (task$task_type == "classif") {
    if (predict_type == "response") {
      response = as.integer(prediction$argmax(dim = 2L))
      class(response) = "factor"
      levels(response) = task$levels(cols = task$target_names)[[1L]]
    } else if (predict_type == "prob") {
      prob = as.numeric(nnf_softmax(prediction, dim = 2L))
      colnames(prob) = task$target_names
    }
    return(list(response = response, prob = prob))
  } else if (task$task_type == "regr") {
    if (predict_type == "response") {
      return(response = as.numeric(prediction))
    } else {
      stopf("Invalid predict_type for task_type 'regr'.")
    }
  } else {
    stopf("Invalid task_type.")
  }

}
