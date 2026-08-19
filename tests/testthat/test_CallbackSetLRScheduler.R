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

test_that("plateau does not step when an epoch has no validation scores", {
  task = tsk("iris")
  mlp = function(...) lrn("classif.mlp",
    callbacks = t_clbk("lr_reduce_on_plateau"), epochs = 8, batch_size = 150, neurons = 10, ...)

  without_valid = mlp()
  expect_no_error(without_valid$train(task))
  expect_equal(without_valid$model$callbacks$lr_reduce_on_plateau$last_epoch, 0)
  expect_equal(without_valid$model$optimizer$param_groups[[1L]]$lr,
    without_valid$param_set$values$opt.lr %??% formals(optim_ignite_adam)$lr)

  every_fourth = mlp(validate = 0.2, measures_valid = msrs("classif.ce"), eval_freq = 4)
  expect_no_error(every_fourth$train(task))
  expect_equal(every_fourth$model$callbacks$lr_reduce_on_plateau$last_epoch, 2)
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


test_that("the scheduler state can be saved and restored", {
  task = tsk("iris")
  make = function(epochs, callbacks) {
    learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
      callbacks = callbacks)
    learner$param_set$set_values(opt.lr = 0.1, cb.lr_step.step_size = 1, cb.lr_step.gamma = 0.5)
    learner
  }

  # a single uninterrupted run of 4 epochs is the reference
  reference = make(4L, t_clbk("lr_step"))
  reference$train(task)

  first = make(2L, t_clbk("lr_step"))
  first$train(task)
  state = first$model$callbacks$lr_step
  expect_equal(state$last_epoch, 2)

  # the state is applied right away when the scheduler already exists, and remembered until
  # $on_begin() otherwise, so the order of the callbacks does not matter
  walk(c(TRUE, FALSE), function(restore_first) {
    restore = torch_callback("restore",
      on_begin = function() self$ctx$callbacks$lr_step$load_state_dict(state))
    cbs = if (restore_first) list(restore, t_clbk("lr_step")) else list(t_clbk("lr_step"), restore)

    second = make(2L, cbs)
    second$train(task)

    # the schedule continued instead of starting over. The learning rate itself is a property of
    # the optimizer, which this run does not carry over, so only the schedule is compared.
    expect_equal(second$model$callbacks$lr_step$last_epoch,
      reference$model$callbacks$lr_step$last_epoch)
    expect_equal(second$model$callbacks$lr_step$.step_count,
      reference$model$callbacks$lr_step$.step_count)
  })
})

test_that("the scheduler state is NULL before the training loop begins", {
  cb = t_clbk("lr_step", step_size = 1)$generate()
  expect_null(cb$state_dict())
})

test_that("loading a one cycle state that was saved for a different schedule errors", {
  task = tsk("iris")
  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("lr_one_cycle", max_lr = 0.1))
  learner$train(task)
  state = learner$model$callbacks$lr_one_cycle

  # the state was saved for a 2-epoch schedule, this run is configured for 4
  restore = torch_callback("restore",
    on_begin = function() self$ctx$callbacks$lr_one_cycle$load_state_dict(state))
  other = lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10,
    callbacks = list(restore, t_clbk("lr_one_cycle", max_lr = 0.1)))

  expect_error(other$train(task), "Cannot load the state of the one cycle")
  # the error is raised before any epoch is trained
  expect_null(other$model)
})

describe("resuming", {
  it("continues the momentum of a schedule that cycles it", {
    # creating the scheduler resets everything it schedules, momentum included, and only the rate
    # is recomputed from `base_lrs` on the next step
    path = tempfile()
    make = function(cbs) lrn("classif.mlp", epochs = 6L, batch_size = 50, neurons = 10, seed = 1,
      callbacks = cbs, cb.lr_one_cycle.max_lr = 0.1)
    crashing_run(path, epochs = 6L, fail_at = 3L, callback = t_clbk("lr_one_cycle"),
      values = list(cb.lr_one_cycle.max_lr = 0.1))

    seen = NULL
    record = function() {
      if (is.null(seen)) seen <<- self$ctx$optimizer$param_groups[[1L]]$betas[[1L]]
    }
    resumed = make(list(t_clbk("lr_one_cycle"), torch_callback("probe", on_batch_begin = record)))
    resumed$param_set$set_values(resume = path)
    resumed$train(tsk("iris"))

    # what an uninterrupted run has at that same step, i.e. the first batch of epoch 3
    reference = NULL
    compare = function() {
      if (self$ctx$global_step == 7L) reference <<- self$ctx$optimizer$param_groups[[1L]]$betas[[1L]]
    }
    make(list(t_clbk("lr_one_cycle"), torch_callback("probe2", on_batch_begin = compare)))$train(tsk("iris"))

    expect_equal(seen, reference)
  })

  it("the learning rate schedule is continued", {
    path = tempfile()
    reference = lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10,
      callbacks = t_clbk("lr_step"))
    reference$param_set$set_values(opt.lr = 0.1, cb.lr_step.step_size = 1, cb.lr_step.gamma = 0.5)
    reference$train(tsk("iris"))

    first = make_checkpoint(epochs = 2L, path = path, callbacks = list(t_clbk("lr_step")),
      opt.lr = 0.1, cb.lr_step.step_size = 1, cb.lr_step.gamma = 0.5)

    resumed = resumer(4L, path, callbacks = t_clbk("lr_step"),
      opt.lr = 0.1, cb.lr_step.step_size = 1, cb.lr_step.gamma = 0.5)
    resumed$train(tsk("iris"))

    # the schedule and the learning rate itself both continue: the schedule comes from the state
    # file, the learning rate from the restored optimizer
    expect_equal(resumed$model$callbacks$lr_step$last_epoch,
      reference$model$callbacks$lr_step$last_epoch)
    expect_equal(resumed$model$optimizer$param_groups[[1L]]$lr,
      reference$model$optimizer$param_groups[[1L]]$lr)
  })

  it("every learning rate scheduler continues its schedule", {
    # the schedulers share `$state_dict()`, but each is configured differently and one cycle
    # additionally rejects a state that was saved for a schedule of a different length
    args = list(
      # T_max is deliberately longer than the run: with T_max = 4 the rate is annealed to `eta_min`
      # by the last epoch in both runs, which every schedule would agree on
      lr_cosine_annealing = list(cb.lr_cosine_annealing.T_max = 10L),
      lr_lambda = list(cb.lr_lambda.lr_lambda = function(epoch) 0.9^epoch),
      lr_multiplicative = list(cb.lr_multiplicative.lr_lambda = function(epoch) 0.9),
      lr_one_cycle = list(cb.lr_one_cycle.max_lr = 0.1),
      lr_step = list(cb.lr_step.step_size = 1, cb.lr_step.gamma = 0.5)
    )

    for (id in names(args)) {
      path = tempfile()
      # what the schedule looks like after four uninterrupted epochs
      reference = lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10, seed = 1,
        callbacks = t_clbk(id))
      reference$param_set$set_values(.values = args[[id]])
      reference$train(tsk("iris"))

      # the same run, but killed in epoch 3 and continued from the checkpoint of epoch 2
      crashing_run(path, epochs = 4L, fail_at = 3L, callback = t_clbk(id), values = args[[id]])
      resumed = lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10, seed = 1,
        resume = path, callbacks = t_clbk(id))
      resumed$param_set$set_values(.values = args[[id]])
      resumed$train(tsk("iris"))

      # the schedule was continued rather than started over, so it is where four epochs leave it.
      # These schedulers step on the epoch (or batch) count alone, so this does not depend on how
      # the two runs happened to see their data.
      state = resumed$model$callbacks[[id]]
      expect_equal(state$last_epoch, reference$model$callbacks[[id]]$last_epoch, info = id)
      expect_equal(state$.last_lr, reference$model$callbacks[[id]]$.last_lr, info = id)
      # and the learning rate the optimizer actually uses came along with it
      expect_equal(resumed$model$optimizer$param_groups[[1L]]$lr,
        reference$model$optimizer$param_groups[[1L]]$lr, info = id)
    }
  })

  it("does not rewind the optimizer's learning rate", {
    # creating a scheduler puts the optimizer back to the `initial_lr` it recorded, which on a
    # resumed run is where the schedule started rather than where it got to. A schedule that
    # computes the next rate from the current one would silently start over from there.
    path = tempfile()
    seen = new.env()
    # weight Inf, so this reads the learning rate after the scheduler callback created its scheduler
    spy = torch_callback("spy", weight = Inf,
      on_begin = function() seen$lr = self$ctx$optimizer$param_groups[[1L]]$lr)

    # lr_multiplicative on purpose: only a scheduler whose constructor rewinds the rate can show
    # this, and of those only a recursive schedule never recovers, because the others recompute the
    # rate from `base_lrs` and `last_epoch` on their next step
    values = list(cb.lr_multiplicative.lr_lambda = function(epoch) 0.9)
    crashing_run(path, epochs = 4L, fail_at = 3L, callback = t_clbk("lr_multiplicative"),
      values = values)
    checkpointed = torch_load(file.path(path, "optimizer2.pt"))$param_groups[[1L]]$lr

    resumed = lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10, seed = 1,
      resume = path, callbacks = list(t_clbk("lr_multiplicative"), spy))
    resumed$param_set$set_values(.values = values)
    resumed$train(tsk("iris"))

    # the schedule had already taken two steps off the initial rate, and that is where the resumed
    # run picks it up rather than at the `initial_lr` of 0.001
    expect_equal(checkpointed, 0.001 * 0.9^2)
    expect_equal(seen$lr, checkpointed)
  })

  it("the reduce-on-plateau schedule continues", {
    # this one steps on the validation score, so only its epoch counter is comparable across runs;
    # what matters is that the best score it has seen is restored rather than reset
    path = tempfile()
    args = list(cb.lr_reduce_on_plateau.patience = 1L, cb.lr_reduce_on_plateau.factor = 0.5)
    task = task_with_valid()
    make = function(...) {
      learner = lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10, seed = 1,
        validate = "predefined", measures_valid = msrs("classif.ce"),
        callbacks = t_clbk("lr_reduce_on_plateau"), ...)
      learner$param_set$set_values(.values = args)
      learner
    }
    crashing_run(path, epochs = 4L, fail_at = 3L, callback = t_clbk("lr_reduce_on_plateau"),
      values = args, task = task, validate = "predefined", measures_valid = msrs("classif.ce"))

    first = readRDS(file.path(path, "state2.rds"))$callbacks$lr_reduce_on_plateau
    resumed = make(resume = path)
    resumed$train(task)

    state = resumed$model$callbacks$lr_reduce_on_plateau
    # a run that started its schedule over would be at 2, the number of epochs it trained itself
    expect_equal(state$last_epoch, 4)
    # the first run's best score is still the one to beat, i.e. the plateau detection was not reset.
    # `best` starts at `mode_worse` (Inf), so check the first run actually recorded one.
    expect_true(is.finite(first$best))
    expect_true(state$best <= first$best)
  })
})
