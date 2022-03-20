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
        y_true = batch$y[, 1L]
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
