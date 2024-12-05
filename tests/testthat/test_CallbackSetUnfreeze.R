test_that("autotest", {
  cb = t_clbk("unfreeze",
    starting_weights = select_all(),
    unfreeze = data.table(epoch = NULL, weights = NULL)
  )
  expect_torch_callback(cb, check_man = TRUE)
})

test_that("unfreezing on epochs works in the end", {
  logger = lgr::get_logger("mlr3")
  buffer = lgr::AppenderBuffer$new()
  logger$add_appender(buffer, name = "buffer")

  task = tsk("iris")
  mlp = lrn("classif.mlp",
            callbacks = t_clbk("unfreeze"),
            cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight", "6.weight", "6.bias"))),
            cb.unfreeze.unfreeze = data.table(
              epoch = c(2, 5),
              weights = list(select_name("0.weight"), select_name(c("3.weight", "6.weight")))
            ),
            epochs = 10, batch_size = 150, neurons = c(100, 200, 300)
  )

  mlp$train(task)

  expect_false(mlp$network$parameters[[select_name("6.bias")(names(mlp$network$parameters))]]$requires_grad)
  expect_true(all(map_lgl(mlp$network$parameters[select_invert(select_name(c("6.bias")))(names(mlp$network$parameters))], function(param) param$requires_grad)))

  expect_length(buffer$data$msg, 3)
  grepl("Freezing the following parameters before training: 0.weight, 3.weight, 6.weight, 6.bias", buffer$data$msg[1], fixed = TRUE)
  grepl("Unfreezing at epoch 2: 0.weight", buffer$data$msg[2], fixed = TRUE)
  grepl("Unfreezing at epoch 5: 3.weight, 6.weight", buffer$data$msg[3], fixed = TRUE)

  buffer$clear()
  logger$remove_appender("buffer")
})

test_that("unfreezing on batches works in the end", {
  logger = lgr::get_logger("mlr3")
  buffer = lgr::AppenderBuffer$new()
  logger$add_appender(buffer, name = "buffer")

  task = tsk("iris")
  mlp = lrn("classif.mlp",
            callbacks = t_clbk("unfreeze"),
            epochs = 10, batch_size = 50, neurons = c(100, 200, 300)
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

  expect_length(buffer$data$msg, 3)
  grepl("Freezing the following parameters before training: 0.weight, 3.weight", buffer$data$msg[1], fixed = TRUE)
  grepl("Unfreezing at batch 2: 0.weight", buffer$data$msg[2], fixed = TRUE)
  grepl("Unfreezing at batch 5: 3.weight", buffer$data$msg[3], fixed = TRUE)

  buffer$clear()
  logger$remove_appender("buffer")
})

test_that("starting weights work", {
  logger = lgr::get_logger("mlr3")
  buffer = lgr::AppenderBuffer$new()
  logger$add_appender(buffer, name = "buffer")

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

  expect_length(buffer$data$msg, 1)
  grepl("Freezing the following parameters before training: 0.weight, 3.weight", buffer$data$msg[1], fixed = TRUE)

  buffer$clear()
  logger$remove_appender("buffer")
})
