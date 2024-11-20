test_that("selectors work", {
  n_epochs = 1

  task = tsk("iris")

  mlp = lrn("classif.mlp",
            epochs = 10, batch_size = 150, neurons = c(100, 200, 300)
  )
  mlp$train(task)

  all_params = names(mlp$network$parameters)

  expect_equal(selectorparam_none()(all_params), character(0))
  expect_equal(selectorparam_all()(all_params), all_params)
  expect_equal(selectorparam_grep("weight")(all_params), c("0.weight", "3.weight", "6.weight", "9.weight"))
  expect_equal(selectorparam_invert(selectorparam_none())(all_params), all_params)
})