test_that("autotest", {
  cb = t_clbk("lr_cosine_annealing", T_max = 10)
  # each LR scheduler has a different paramset, so we don't test them
  expect_torch_callback(cb, check_paramset = FALSE)
})

test_that("decay works", {
  cb = t_clbk("lr_step")
  task = tsk("iris")
  n_epochs = 10

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )
  gamma = 0.5
  step_size = 2

  mlp$param_set$set_values(cb.lr_step.gamma = gamma)
  mlp$param_set$set_values(cb.lr_step.step_size = step_size)

  mlp$train(task)

  expect_equal(mlp$model$optimizer$param_groups[[1]]$initial_lr * gamma^(n_epochs / step_size),
               mlp$model$optimizer$param_groups[[1]]$lr)
})

test_that("custom LR scheduler works", {
  # modeled after lr_step
  lr_subtract <- lr_scheduler(
    "lr_subtract",
    initialize = function(optimizer, step_size, delta = 0.1, last_epoch = -1) {
      self$step_size <- step_size
      self$delta <- delta
      super$initialize(optimizer, last_epoch)
    },
    get_lr = function() {
      if ((self$last_epoch == 0) || (self$last_epoch %% self$step_size != 0)) {
        return(sapply(self$optimizer$param_groups, function(x) x$lr))
      }

      sapply(self$optimizer$param_groups, function(x) x$lr - self$delta)
    }
  )
  cb = as_lr_scheduler(lr_subtract, step_on_epoch = TRUE)
  expect_torch_callback(cb, check_paramset = FALSE)

  task = tsk("iris")
  n_epochs = 10

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )
  reduction_amt = 0.00001
  step_size = 2
  mlp$param_set$set_values(cb.lr_subtract.delta = reduction_amt)
  mlp$param_set$set_values(cb.lr_subtract.step_size = step_size)

  mlp$train(task)

  expect_equal(mlp$model$optimizer$param_groups[[1]]$initial_lr - ((n_epochs / step_size) * reduction_amt),
               mlp$model$optimizer$param_groups[[1]]$lr)
})
