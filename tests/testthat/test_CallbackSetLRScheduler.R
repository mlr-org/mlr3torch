test_that("autotest", {
  cb_ca = t_clbk("lr_cosine_annealing", T_max = 10)
  # each LR scheduler has a different paramset, so we don't test them
  expect_torch_callback(cb_ca, check_paramset = FALSE)

  lambda1 <- function(epoch) epoch %/% 30
  lambda2 <- function(epoch) 0.95^epoch
  cb_lambda = t_clbk("lr_lambda", lr_lambda = list(lambda1, lambda2))
  expect_torch_callback(cb_lambda, check_paramset = FALSE)

  lambda <- function(epoch) 0.95
  cb_mult = t_clbk("lr_multiplicative", lr_lambda = lambda)
  expect_torch_callback(cb_mult, check_paramset = FALSE)

  cb_1cycle = t_clbk("lr_one_cycle", max_lr = 0.1)
  expect_torch_callback(cb_1cycle, check_paramset = FALSE)

  cb_plateau = t_clbk("lr_reduce_on_plateau")
  expect_torch_callback(cb_plateau, check_paramset = FALSE)

  cb_step = t_clbk("lr_step", step_size = 4)
  expect_torch_callback(cb_step, check_paramset = FALSE)
})

test_that("cosine annealing works", {
  cb = t_clbk("lr_cosine_annealing")
  task = tsk("iris")

  n_epochs = 10

  mlp = lrn("classif.mlp",
    callbacks = cb,
    epochs = n_epochs, batch_size = 150, neurons = 10,
    measures_train = msrs(c("classif.acc", "classif.ce"))
  )

  T_max = 2
  eta_min = 0.0001
  mlp$param_set$set_values(cb.lr_cosine_annealing.T_max = T_max)
  mlp$param_set$set_values(cb.lr_cosine_annealing.eta_min = eta_min)

  mlp$train(task)

  expect_equal(eta_min, mlp$model$optimizer$param_groups[[1]]$lr)
})

test_that("lambda works", {
  cb = t_clbk("lr_lambda")
  task = tsk("iris")

  n_epochs = 10

  mlp = lrn("classif.mlp",
    callbacks = cb,
    epochs = n_epochs, batch_size = 150, neurons = 10,
    measures_train = msrs(c("classif.acc", "classif.ce"))
  )

  lambda1 <- function(epoch) 0.95 ^ epoch
  mlp$param_set$set_values(cb.lr_lambda.lr_lambda = list(lambda1))

  mlp$train(task)

  expect_equal(mlp$model$optimizer$param_groups[[1]]$initial_lr * 0.95^(n_epochs),
              mlp$model$optimizer$param_groups[[1]]$lr)
})

test_that("multiplicative works", {
  cb = t_clbk("lr_multiplicative")
  task = tsk("iris")

  n_epochs = 10

  mlp = lrn("classif.mlp",
    callbacks = cb,
    epochs = n_epochs, batch_size = 150, neurons = 10,
    measures_train = msrs(c("classif.acc", "classif.ce"))
  )

  lambda <- function(epoch) 0.95
  mlp$param_set$set_values(cb.lr_multiplicative.lr_lambda = lambda)

  mlp$train(task)

  expect_equal(mlp$model$optimizer$param_groups[[1]]$initial_lr * 0.95^(n_epochs),
              mlp$model$optimizer$param_groups[[1]]$lr)
})

test_that("step decay works", {
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

test_that("plateau works", {
  cb = t_clbk("lr_reduce_on_plateau")

  task = tsk("iris")

  mlp = lrn("classif.mlp",
    callbacks = cb,
    epochs = 10, batch_size = 150, neurons = 10,
    measures_train = msrs(c("classif.acc", "classif.ce")),
    measures_valid = msrs(c("classif.ce")),
    validate = 0.2
  )

  mlp$param_set$set_values(cb.lr_reduce_on_plateau.mode = "min")

  mlp$train(task)

  expect_learner(mlp)
  expect_class(mlp$network, c("nn_sequential", "nn_module"))
})

test_that("1cycle works", {
  cb = t_clbk("lr_one_cycle", max_lr = 0.01)

  task = tsk("iris")

  mlp = lrn("classif.mlp",
    callbacks = cb,
    epochs = 10, batch_size = 50, neurons = 10,
    measures_train = msrs(c("classif.acc", "classif.ce"))
  )

  mlp$train(task)

  expect_learner(mlp)
  expect_class(mlp$network, c("nn_sequential", "nn_module"))
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
