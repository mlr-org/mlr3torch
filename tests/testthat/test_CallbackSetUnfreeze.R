test_that("autotest", {
  cb = t_clbk("unfreeze")
  expect_torch_callback(cb, check_man = TRUE)
})

test_that("weights are frozen correctly using epochs", {
  cb = t_clbk("unfreeze")
  n_epochs = 10

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            validate = 0.2,
            measures_valid = msrs(c("classif.acc", "classif.ce")),
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )

  # begin LLM
  # Test with simple layer selection
  mlp$param_set$set_values(
    cb.freeze.starting_weights = selector_name("layer1"),
    cb.freeze.unfreeze = data.table(
      weights = list(selector_name("layer2")), 
      epochs = 2
    )
  )
  
  # Verify initial frozen state
  expect_false(mlp$model$layer2$requires_grad)
  expect_true(mlp$model$layer1$requires_grad)
  
  # Train for 3 epochs
  mlp$train(task)
  expect_true(mlp$model$layer2$requires_grad)
  # end LLM
})

test_that("weights are frozen correctly using batches", {
  cb = t_clbk("unfreeze")
  n_epochs = 10

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            validate = 0.2,
            measures_valid = msrs(c("classif.acc", "classif.ce")),
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )

  # begin LLM
  # Test with multiple layer unfreezing
  mlp$param_set$set_values(
    cb.freeze.starting_weights = selector_none(),
    cb.freeze.unfreeze = data.table(
      weights = list(
        selector_name("layer1"),
        selector_name("layer2")
      ),
      batches = c(10, 20)
    )
  )
  
  # Verify all layers start frozen
  expect_false(mlp$model$layer1$requires_grad)
  expect_false(mlp$model$layer2$requires_grad)
  
  mlp$train(task)
  expect_true(mlp$model$layer1$requires_grad)
  expect_true(mlp$model$layer2$requires_grad)
  # end LLM
})

# TODO: decide whether we want to test this (Copilot suggestion)
test_that("invalid configurations throw errors", {
  cb = t_clbk("unfreeze")
  
  expect_error(
    cb$param_set$set_values(
      starting_weights = "invalid_selector",
      unfreeze = data.table(weights = list(), epochs = numeric())
    ),
    "must be a Selector"
  )
  
  expect_error(
    cb$param_set$set_values(
      starting_weights = selector_none(),
      unfreeze = data.table(weights = list(), invalid_column = numeric())
    ),
    "must contain either 'epochs' or 'batches'"
  )
})

# TODO: decide whether we want to test this (Copilot suggestion)
test_that("gradual unfreezing works correctly", {
  cb = t_clbk("unfreeze")
  n_epochs = 5

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            validate = 0.2,
            measures_valid = msrs(c("classif.acc", "classif.ce")),
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )

  mlp$param_set$set_values(
    cb.freeze.starting_weights = selector_none(),
    cb.freeze.unfreeze = data.table(
      weights = list(
        selector_name("layer1"),
        selector_name("layer2"),
        selector_name("layer3")
      ),
      epochs = c(1, 2, 3)
    )
  )

  # Check initial state
  expect_false(mlp$model$layer1$requires_grad)
  expect_false(mlp$model$layer2$requires_grad)
  expect_false(mlp$model$layer3$requires_grad)
  
  # Train and check progressive unfreezing
  mlp$train(task)
  expect_true(mlp$model$layer1$requires_grad)
  expect_true(mlp$model$layer2$requires_grad)
  expect_true(mlp$model$layer3$requires_grad)
})