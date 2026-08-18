test_that("autotest", {
  cb = t_clbk("unfreeze",
    starting_weights = select_all(),
    unfreeze = data.table(epoch = NULL, weights = NULL)
  )
  expect_torch_callback(cb, check_man = TRUE)
})

test_that("unfreezing on epochs works in the end", {
  task = tsk("iris")
  mlp = lrn("classif.mlp",
            callbacks = t_clbk("unfreeze"),
            cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight", "6.weight", "6.bias"))),
            cb.unfreeze.unfreeze = data.table(
              epoch = c(2, 5),
              weights = list(select_name("0.weight"), select_name(c("3.weight", "6.weight")))
            ),
            epochs = 6, batch_size = 150, neurons = c(1, 1, 1)
  )

  mlp$train(task)

  expect_false(mlp$network$parameters[[select_name("6.bias")(names(mlp$network$parameters))]]$requires_grad)
  expect_true(all(map_lgl(mlp$network$parameters[select_invert(select_name(c("6.bias")))(names(mlp$network$parameters))], function(param) param$requires_grad)))
})

test_that("unfreezing on batches works in the end", {
  task = tsk("iris")
  mlp = lrn("classif.mlp",
            callbacks = t_clbk("unfreeze"),
            epochs = 10, batch_size = 50, neurons = c(1, 1, 1)
  )

  mlp$param_set$set_values(cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight", "6.weight"))))

  mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
      batch = c(2, 5),
      weights = list(select_name("0.weight"), select_name("3.weight"))
    )
  )

  mlp$train(task)

  expect_false(mlp$network$parameters[[select_name("6.weight")(names(mlp$network$parameters))]]$requires_grad)
  expect_true(all(map_lgl(mlp$network$parameters[select_invert(select_name(c("6.weight")))(names(mlp$network$parameters))], function(param) param$requires_grad)))
})

test_that("starting weights work", {
  task = tsk("iris")
  mlp = lrn("classif.mlp",
            callbacks = t_clbk("unfreeze"),
            cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight"))),
            cb.unfreeze.unfreeze = data.table(),
            epochs = 2, batch_size = 150, neurons = c(100, 200, 300)
  )

  mlp$train(task)

  expect_false(mlp$network$parameters[[select_name("0.weight")(names(mlp$network$parameters))]]$requires_grad)
  expect_false(mlp$network$parameters[[select_name("3.weight")(names(mlp$network$parameters))]]$requires_grad)
  expect_true(all(map_lgl(mlp$network$parameters[select_invert(select_name(c("0.weight", "3.weight")))(names(mlp$network$parameters))], function(param) param$requires_grad)))
})

test_that("input checks work", {
  expect_error(t_clbk("unfreeze", starting_weights = 123), "Select")
  expect_error(t_clbk("unfreeze", starting_weights = select_name("a"), unfreeze = 1L), "data.table")
  expect_error(t_clbk("unfreeze", starting_weights = select_name("a"), unfreeze = data.table(
    weights = list(select_all()), batch = "a")), "integerish")
  expect_error(t_clbk("unfreeze", starting_weights = select_name("a"), unfreeze = data.table(
    weights = list(select_all(), select_all()), batch = c(1L, 1L))), "duplicates")
  expect_error(t_clbk("unfreeze", starting_weights = select_name("a"), unfreeze = data.table(
    weights = list(select_all(), select_all()), batch = c(1L, 2L))), NA)
  expect_error(t_clbk("unfreeze", starting_weights = select_name("a"), unfreeze = data.table(
    weights = list(select_all()), batch = 1L)), NA)
  expect_error(t_clbk("unfreeze", starting_weights = select_name("a"), unfreeze = data.table(
    weights = list(select_all()), epoch = 1L)), NA)
})

test_that("the set of trainable weights can be saved and restored", {
  task = tsk("iris")
  make = function(epochs, callbacks) {
    lrn("classif.mlp", epochs = epochs, batch_size = 150, neurons = c(1, 1, 1),
      callbacks = callbacks,
      cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight"))),
      cb.unfreeze.unfreeze = data.table(epoch = 2, weights = list(select_name("0.weight")))
    )
  }

  first = make(2L, t_clbk("unfreeze"))
  first$train(task)
  state = first$model$callbacks$unfreeze
  expect_true("0.weight" %in% state$trainable)
  expect_true("3.weight" %nin% state$trainable)

  # epoch 2 is not reached again, so '0.weight' would be frozen again without the restored state.
  # The state is restored regardless of whether it is loaded before or after $on_begin() of the
  # unfreeze callback froze the network according to `starting_weights`.
  restore = torch_callback("restore",
    on_begin = function() self$ctx$callbacks$unfreeze$load_state_dict(state))
  walk(list(list(restore, t_clbk("unfreeze")), list(t_clbk("unfreeze"), restore)), function(cbs) {
    second = make(1L, cbs)
    second$train(task)
    expect_true("0.weight" %in% second$model$callbacks$unfreeze$trainable)
    expect_true("3.weight" %nin% second$model$callbacks$unfreeze$trainable)
  })
})


test_that("restored trainable weights that the network does not have are ignored", {
  # the resuming run builds its network from its own configuration, so a state can name weights
  # that network does not have -- e.g. after a change of task or of `neurons`. Those must be
  # skipped rather than stop the run from starting.
  task = tsk("iris")
  state = list(trainable = c("0.weight", "99.weight"))
  restore = torch_callback("restore",
    on_begin = function() self$ctx$callbacks$unfreeze$load_state_dict(state))

  learner = lrn("classif.mlp", epochs = 1L, batch_size = 150, neurons = c(1, 1, 1),
    callbacks = list(restore, t_clbk("unfreeze")),
    cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight"))),
    cb.unfreeze.unfreeze = data.table(epoch = 2, weights = list(select_name("0.weight")))
  )
  expect_no_error(learner$train(task))

  trainable = learner$model$callbacks$unfreeze$trainable
  # the weight the network does have was unfrozen, the one it does not have was passed over
  expect_true("0.weight" %in% trainable)
  expect_true("99.weight" %nin% trainable)
  # and a weight the state did not name is still frozen
  expect_true("3.weight" %nin% trainable)
})
