check_frozen = torch_callback("custom_logger",
  initialize = function(alpha = 0.1) {
    self$starting_weights = starting_weights
    # consider supporting character vectors
    self$unfreeze = unfreeze
  },
  on_epoch_end = function() {
    if (self$ctx$epoch %in% self$unfreeze$epoch) {
      weights = (self$unfreeze[epoch == self$ctx$epoch]$unfreeze)[[1]](names(self$ctx$network$parameters))
      print(self$ctx$network$parameters[weights]$requires_grad)
    }
  },
  on_batch_end = function() {
    batch_num = (self$ctx$epoch - 1) * length(self$ctx$loader_train) + self$ctx$step
    if (batch_num %in% self$unfreeze$batch) {
      weights = (self$unfreeze[batch == batch_num]$unfreeze)[[1]](names(self$ctx$network$parameters))
      print(self$ctx$network$parameters[weights]$requires_grad)
    }
  }
)

test_that("autotest", {
  cb = t_clbk("unfreeze", 
    starting_weights = select_all(),
    unfreeze = data.table()
  )
  expect_torch_callback(cb, check_man = TRUE)
})

test_that("unfreezing on epochs works in the end", {
  n_epochs = 10

  task = tsk("iris")

  mlp = lrn("classif.mlp",
            callbacks = t_clbk("unfreeze"),
            epochs = 10, batch_size = 150, neurons = c(100, 200, 300)
  )

  mlp$param_set$set_values(cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight"))))

  mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
      epoch = 2,
      unfreeze = select_name("0.weight")
    )
  )

  mlp$train(task)

  expect_true(mlp$network$parameters[[select_name("0.weight")(names(mlp$network$parameters))]]$requires_grad)
  expect_false(mlp$network$parameters[[select_name("3.weight")(names(mlp$network$parameters))]]$requires_grad)
})

test_that("unfreezing on batches works in the end", {
  n_epochs = 10

  task = tsk("iris")

  mlp = lrn("classif.mlp",
            callbacks = t_clbk("unfreeze"),
            epochs = 10, batch_size = 50, neurons = c(100, 200, 300)
  )

  mlp$param_set$set_values(cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight"))))

  mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
      batch = 2,
      unfreeze = select_name("0.weight")
    )
  )

  mlp$train(task)

  expect_true(mlp$network$parameters[[select_name("0.weight")(names(mlp$network$parameters))]]$requires_grad)
  expect_false(mlp$network$parameters[[select_name("3.weight")(names(mlp$network$parameters))]]$requires_grad)
})

test_that("freezing both with batches and epochs works at the correct time", {
  n_epochs = 10

  task = tsk("iris")

  mlp = lrn("classif.mlp",
            callbacks = list(t_clbk("unfreeze"), check_frozen),
            epochs = 10, batch_size = 150, neurons = c(100, 200, 300)
  )

  # mlp$param_set$set_values(cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight"))))

  # mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
  #     epoch = 2,
  #     unfreeze = select_name("0.weight")
  #   )
  # )

  # mlp$train(task)

  # expect_true(mlp$network$parameters[[select_name("0.weight")(names(mlp$network$parameters))]]$requires_grad)
  # expect_false(mlp$network$parameters[[select_name("3.weight")(names(mlp$network$parameters))]]$requires_grad)
})


# # realistic example using epochs
# # TODO: write a custom callback that accesses the requires_grad of a parameter
# # you can actually write a test for this callback as well
# # such as: 
# test_that("weights are frozen correctly using epochs", {
#   n_epochs = 10

#   task = tsk("iris")

#   mlp = lrn("classif.mlp",
#             callbacks = t_clbk("unfreeze"),
#             epochs = 10, batch_size = 150, neurons = c(100, 200, 300)
#   )

#   mlp$param_set$set_values(cb.unfreeze.starting_weights = selectorparam_all())
#   # mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
#   #   epoch = c(2, 4),
#   #   unfreeze = list(selectorparam_grep("9"), selectorparam_))
#   # )
#   mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table())

#   mlp$train(task)

#   expect_true(all(mlr3misc::map_lgl(mlp$network$parameters, function(param) param$requires_grad)))

#   # # begin LLM
#   # # Test with simple layer selection
#   # mlp$param_set$set_values(
#   #   cb.unfreeze.starting_weights = selectorparam_name(c("9.weight", "9.bias")),
#   #   cb.unfreeze.unfreeze = data.table(
#   #     weights = list(selectorparam_name("layer2")), 
#   #     epochs = 2
#   #   )
#   # )
  
#   # # Verify initial frozen state
#   # expect_false(mlp$model$layer2$requires_grad)
#   # expect_true(mlp$model$layer1$requires_grad)
  
#   # # Train for 3 epochs
#   # mlp$train(task)
#   # expect_true(mlp$model$layer2$requires_grad)
#   # # end LLM
# })

# test_that("weights are frozen correctly using batches", {
#   cb = t_clbk("unfreeze")
#   n_epochs = 10

#   mlp = lrn("classif.mlp",
#             callbacks = t_clbk("tb"),
#             epochs = 10, batch_size = 150, neurons = c(100, 200, 300)
#   )

#   # # begin LLM
#   # # Test with multiple layer unfreezing
#   # mlp$param_set$set_values(
#   #   cb.unfreeze.starting_weights = selector_none(),
#   #   cb.unfreeze.unfreeze = data.table(
#   #     weights = list(
#   #       selector_name("layer1"),
#   #       selector_name("layer2")
#   #     ),
#   #     batches = c(10, 20)
#   #   )
#   # )
  
#   # # Verify all layers start frozen
#   # expect_false(mlp$model$layer1$requires_grad)
#   # expect_false(mlp$model$layer2$requires_grad)
  
#   # mlp$train(task)
#   # expect_true(mlp$model$layer1$requires_grad)
#   # expect_true(mlp$model$layer2$requires_grad)
#   # # end LLM
# })

# # TODO: decide whether we want to test this (Copilot suggestion)
# test_that("invalid configurations throw errors", {
#   cb = t_clbk("unfreeze")
  
#   expect_error(
#     cb$param_set$set_values(
#       starting_weights = "invalid_selector",
#       unfreeze = data.table(weights = list(), epochs = numeric())
#     ),
#     "must be a Selector"
#   )
  
#   expect_error(
#     cb$param_set$set_values(
#       starting_weights = selector_none(),
#       unfreeze = data.table(weights = list(), invalid_column = numeric())
#     ),
#     "must contain either 'epochs' or 'batches'"
#   )
# })

# # TODO: decide whether we want to test this (Copilot suggestion)
# test_that("gradual unfreezing works correctly", {
#   cb = t_clbk("unfreeze")
#   n_epochs = 5

#   mlp = lrn("classif.mlp",
#             callbacks = cb,
#             epochs = n_epochs, batch_size = 150, neurons = 10,
#             validate = 0.2,
#             measures_valid = msrs(c("classif.acc", "classif.ce")),
#             measures_train = msrs(c("classif.acc", "classif.ce"))
#   )

#   mlp$param_set$set_values(
#     cb.unfreeze.starting_weights = selector_none(),
#     cb.unfreeze.unfreeze = data.table(
#       weights = list(
#         selector_name("layer1"),
#         selector_name("layer2"),
#         selector_name("layer3")
#       ),
#       epochs = c(1, 2, 3)
#     )
#   )

#   # Check initial state
#   expect_false(mlp$model$layer1$requires_grad)
#   expect_false(mlp$model$layer2$requires_grad)
#   expect_false(mlp$model$layer3$requires_grad)
  
#   # Train and check progressive unfreezing
#   mlp$train(task)
#   expect_true(mlp$model$layer1$requires_grad)
#   expect_true(mlp$model$layer2$requires_grad)
#   expect_true(mlp$model$layer3$requires_grad)
# })