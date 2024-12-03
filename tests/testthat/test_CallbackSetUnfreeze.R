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
            cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight", "6.weight"))),
            cb.unfreeze.unfreeze = data.table(
              epoch = c(2, 5),
              weights = list(select_name("0.weight"), select_name("3.weight"))
            ),
            epochs = 10, batch_size = 150, neurons = c(100, 200, 300)
  )

  # mlp$param_set$set_values(cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight", "6.weight"))))
  # mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
  #     epoch = c(2, 5),
  #     weights = list(select_name(c("0.weight", "3.weight")))
  #   )
  # )

  mlp$train(task)

  expect_true(mlp$network$parameters[[select_name("0.weight")(names(mlp$network$parameters))]]$requires_grad)
  expect_false(mlp$network$parameters[[select_name("3.weight")(names(mlp$network$parameters))]]$requires_grad)
  expect_true(mlp$network$parameters[[select_invert(select_name(c("0.weight", "3.weight")))(names(mlp$network$parameters))]]$requires_grad)
})

test_that("unfreezing on batches works in the end", {
  task = tsk("iris")
  mlp = lrn("classif.mlp",
            callbacks = t_clbk("unfreeze"),
            epochs = 10, batch_size = 50, neurons = c(100, 200, 300)
  )

  mlp$param_set$set_values(cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight", "6.weight"))))

  mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
      batch = c(2, 5),
      weights = list(select_name(c("0.weight", "3.weight")))
    )
  )

  mlp$train(task)

  expect_true(mlp$network$parameters[[select_name("0.weight")(names(mlp$network$parameters))]]$requires_grad)
  expect_false(mlp$network$parameters[[select_name("3.weight")(names(mlp$network$parameters))]]$requires_grad)
  expect_true(mlp$network$parameters[[select_invert(select_name(c("0.weight", "3.weight")))(names(mlp$network$parameters))]]$requires_grad)
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

  print(names(mlp$network$parameters))

  expect_false(mlp$network$parameters[[select_name(c("0.weight", "3.weight"))(names(mlp$network$parameters))]]$requires_grad)
  expect_true(mlp$network$parameters[[select_invert(select_name(c("0.weight", "3.weight")))(names(mlp$network$parameters))]]$requires_grad)
})
