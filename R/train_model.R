predict_torch = function(model, task, device, batch_size) {
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

train_torch = function(model, task, optimizer, criterion, n_epochs, batch_size, device) {
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
        y_true = batch$y[, 1]
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
