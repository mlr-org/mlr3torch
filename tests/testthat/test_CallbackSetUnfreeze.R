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
    weights = list(select_all(), select_all()), batch = c(1L, 2L))), "duplicates")
  expect_error(t_clbk("unfreeze", starting_weights = select_name("a"), unfreeze = data.table(
    weights = list(select_all()), batch = 1L)), NA)
  expect_error(t_clbk("unfreeze", starting_weights = select_name("a"), unfreeze = data.table(
    weights = list(select_all()), epoch = 1L)), NA)
})
