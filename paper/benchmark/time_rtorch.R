time_rtorch = function(epochs, batch_size, n_layers, latent, n, p, device, jit, seed, optimizer, mlr3torch = FALSE) {
  library(mlr3torch)
  library(torch)
  torch_set_num_threads(1)
  torch_manual_seed(seed)

  lr = 0.0001

  make_network = function(p, latent, n_layers) {
    if (n_layers == 0) return(nn_linear(p, 1))
    layers = list(nn_linear(p, latent), nn_relu())
    for (i in seq_len(n_layers - 1)) {
        layers = c(layers, list(nn_linear(latent, latent), nn_relu()))
    }
    layers = c(layers, list(nn_linear(latent, 1)))

    net = do.call(nn_sequential, args = layers)
    net
  }


  X = torch_randn(n, p, device = device)
  beta = torch_randn(p, 1, device = device)
  Y = X$matmul(beta) + torch_randn(n, 1, device = device) * 0.1^2

  net = make_network(p, latent, n_layers)

  opt_class = switch(optimizer,
    "ignite_adamw" = optim_ignite_adamw,
    "adamw" = optim_adamw,
    "sgd" = optim_sgd,
    "ignite_sgd" = optim_ignite_sgd
  )


  loss_fn = nn_mse_loss()
  if (jit && !mlr3torch) {
    net = jit_trace(net, torch_randn(1, p), respect_mode = TRUE)
  }

  net$to(device = device)

  steps = ceiling(n / batch_size)

  dataset = torch::dataset(
    initialize = function(X, Y) {
      self$X = X
      self$Y = Y
    },
    .getbatch = function(i) {
      list(x = self$X[i, drop = FALSE], y = self$Y[i, drop = FALSE])
    },
    .length = function() {
      nrow(self$X)
    }
  )(X, Y)


  # this function should train the network for the given number of epochs and return the final training loss
  train_run_torch = {
    do_step = function(input, target, opt) {
      opt$zero_grad()
      loss = loss_fn(net$forward(input), target)
      loss$backward()
      opt$step()
      loss$item()
    }


    function(epochs) {
      opt = opt_class(net$parameters, lr = lr)
      dataloader = torch::dataloader(dataset, batch_size = batch_size, shuffle = FALSE)
      t0 = Sys.time()
      for (epoch in seq(epochs)) {
        step = 0
        iter = dataloader_make_iter(dataloader)
        while (step < length(dataloader)) {
          batch = dataloader_next(iter)
          y_hat = net$forward(batch$x)
          opt$zero_grad()
          loss = loss_fn(y_hat, batch$y)
          loss$backward()
          opt$step()
          step = step + 1
        }
      }
      as.numeric(difftime(Sys.time(), t0, units = "secs"))
    }
  }

  train_run_mlr3torch = {
    learner = LearnerTorchModel$new(
      task_type = "regr",
      optimizer = as_torch_optimizer(opt_class),
      ingress_tokens = list(x = ingress_ltnsr()),
      loss = as_torch_loss(nn_mse_loss)
    )
    timer = torch_callback("timer",
      on_begin = function() {
        self$t0 = Sys.time()
      },
      on_end = function() {
        self$t1 = Sys.time()
      },
      state_dict = function() {
        c(self$t0, self$t1)
      },
      load_state_dict = function(state_dict) {
        NULL # not needed here
      }
    )

    learner$configure(
      opt.lr = lr,
      device = device,
      drop_last = FALSE,
      batch_size = batch_size,
      shuffle = FALSE,
      callbacks = timer,
      jit_trace = jit,
      tensor_dataset = "device"
    )

    task = as_task_regr(data.table(
      x = as_lazy_tensor(X),
      y = as.numeric(Y)
    ), target = "y")


    function(epochs) {
      learner$.__enclos_env__$private$.network_stored = net
      learner$configure(epochs = epochs)
      learner$train(task)
      ts = learner$model$callbacks$timer
      as.numeric(difftime(ts[2], ts[1], units = "secs"))
    }

  }

  eval_run = function() {
      #net$eval()
      mean_loss = 0
      with_no_grad({
        dataloader = torch::dataloader(dataset, batch_size = batch_size, shuffle = FALSE)
        coro::loop(for (batch in dataloader) {
          y_hat = net$forward(batch[[1]])
          loss = loss_fn(y_hat, batch[[2]])
          mean_loss = mean_loss + loss$item()
        })
      })
      mean_loss / steps
  }
  # warmup
  # If we don't run the same warump code we get weird artifcats
  # because one has more RAM than the other
  train_run_torch(2)
  train_run_mlr3torch(2)

  if (device == "cuda") cuda_synchronize()
  time = if (mlr3torch) {
    train_run_mlr3torch(epochs)
  } else {
    train_run_torch(epochs)
  }
  if (device == "cuda") cuda_synchronize()

  cuda_memory = if (device == "cuda") {
    stats = cuda_memory_stats()
    stats$reserved_bytes$all$current
  } else {
    NA
  }

  list(time = time, loss = eval_run(), cuda_memory = cuda_memory)
}
