test_that("autotest", {
  cb = t_clbk("lr_scheduler_cosine_annealing", T_max = 1)
  expect_torch_callback(cb)
})

test_that("decay works", {
  cb = t_clbk("lr_scheduler_step")
  task = tsk("iris")
  n_epochs = 10

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )
  mlp$param_set$set_values(cb.lr_scheduler.gamma = 0.5)
  mlp$param_set$set_values(cb.lr_scheduler.step_size = 2)

  mlp$train(task)

  expect_equal(mlp$model$optimizer$param_groups[[1]]$initial_lr * (0.5)^(n_epochs / 2),
               mlp$model$optimizer$param_groups[[1]]$lr)
})

test_that("custom LR scheduler works", {
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
  cb = as_lr_scheduler(lr_subtract)
  expect_torch_callback(cb)

  task = tsk("iris")
  n_epochs = 10

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )
  reduction_amt = 0.00001
  mlp$param_set$set_values(cb.lr_scheduler.delta = reduction_amt)
  mlp$param_set$set_values(cb.lr_scheduler.step_size = 2)

  mlp$train(task)

  expect_equal(mlp$model$optimizer$param_groups[[1]]$initial_lr - ((n_epoch / x) * reduction_amt),
               mlp$model$optimizer$param_groups[[1]]$lr)
})
