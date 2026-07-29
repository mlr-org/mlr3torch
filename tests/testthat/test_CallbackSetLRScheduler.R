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


test_that("the scheduler state is restored when resuming", {
  task = tsk("iris")
  path = tempfile()
  make = function(epochs, callbacks) {
    learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
      callbacks = callbacks)
    learner$param_set$set_values(opt.lr = 0.1, cb.lr_step.step_size = 1, cb.lr_step.gamma = 0.5)
    learner
  }

  # a single uninterrupted run of 4 epochs is the reference
  reference = make(4L, t_clbk("lr_step"))
  reference$train(task)

  learner = make(2L, list(t_clbk("lr_step"), t_clbk("checkpoint", freq = 1)))
  learner$param_set$set_values(cb.checkpoint.path = path)
  learner$train(task)

  resumed = make(4L, list(t_clbk("resume"), t_clbk("lr_step")))
  resumed$param_set$set_values(cb.resume.path = path)
  resumed$train(task)

  # without the restored scheduler state the schedule would start over and the lr would be too high
  expect_equal(
    resumed$model$optimizer$param_groups[[1]]$lr,
    reference$model$optimizer$param_groups[[1]]$lr
  )
})

test_that("lr_one_cycle and lr_reduce_on_plateau also restore their state", {
  task = tsk("iris")
  # a run that is interrupted after 2 of 4 epochs, as happens when a job hits a time limit.
  # The resumed run is configured with the same `epochs`, which matters for schedules such as
  # lr_one_cycle that are defined over the total number of steps.
  cb_interrupt = torch_callback("Interrupt",
    on_epoch_end = function() if (self$ctx$epoch >= 2L) self$ctx$terminate = TRUE
  )

  walk(c("lr_one_cycle", "lr_reduce_on_plateau"), function(id) {
    path = tempfile()
    args = if (id == "lr_one_cycle") list(max_lr = 0.1) else list()
    make = function(callbacks) {
      lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10,
        validate = 0.3, measures_valid = msrs("classif.acc"), callbacks = callbacks)
    }

    learner = make(list(invoke(t_clbk, .args = c(list(id), args)), t_clbk("checkpoint", freq = 1),
      cb_interrupt))
    learner$param_set$set_values(cb.checkpoint.path = path)
    learner$train(task)
    expect_equal(learner$model$epochs, 2L)

    state = readRDS(file.path(path, "state2.rds"))
    expect_names(names(state$callbacks), must.include = id)
    expect_equal(state$callbacks[[id]]$last_epoch, learner$model$callbacks[[id]]$last_epoch)

    resumed = make(list(t_clbk("resume"), invoke(t_clbk, .args = c(list(id), args))))
    resumed$param_set$set_values(cb.resume.path = path)
    resumed$train(task)
    expect_equal(resumed$model$epochs, 4L)
    # the schedule continued instead of starting over
    expect_gt(resumed$model$callbacks[[id]]$last_epoch, state$callbacks[[id]]$last_epoch)
  })
})

test_that("resuming lr_one_cycle with a different number of epochs errors immediately", {
  task = tsk("iris")
  path = tempfile()
  interrupt = torch_callback("Interrupt",
    on_epoch_end = function() if (self$ctx$epoch >= 2L) self$ctx$terminate = TRUE)

  learner = lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("lr_one_cycle", max_lr = 0.1), t_clbk("checkpoint", freq = 1), interrupt))
  learner$param_set$set_values(cb.checkpoint.path = path)
  learner$train(task)

  # the checkpoint was written for a 4-epoch schedule
  resumed = lrn("classif.mlp", epochs = 6L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("resume"), t_clbk("lr_one_cycle", max_lr = 0.1)))
  resumed$param_set$set_values(cb.resume.path = path)
  expect_error(resumed$train(task), "Cannot resume the one cycle learning rate schedule")

  # the error is raised before any epoch is trained, not somewhere in the middle of the run
  expect_null(resumed$model)
  expect_set_equal(list.files(path),
    c(paste0("network", 1:2, ".pt"), paste0("optimizer", 1:2, ".pt"), paste0("state", 1:2, ".rds")))
})
