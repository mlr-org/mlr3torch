test_that("autotest", {
  cb = t_clbk("freeze")
  expect_torch_callback(cb, check_man = TRUE)
})

test_that("weights are frozen correctly using epochs", {
  cb = t_clbk("freeze")

  n_epochs = 10

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            validate = 0.2,
            measures_valid = msrs(c("classif.acc", "classif.ce")),
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )

  mlp$param_set$set_values(cb.freeze.starting_weights = ...)
  mlp$param_set$set_values(cb.freeze.unfreeze = data.table(weights = ..., epochs = ...))
})

test_that("weights are frozen correctly using batches", {
  cb = t_clbk("freeze")

  n_epochs = 10

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            validate = 0.2,
            measures_valid = msrs(c("classif.acc", "classif.ce")),
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )

  mlp$param_set$set_values(cb.freeze.starting_weights = ...)
  mlp$param_set$set_values(cb.freeze.unfreeze = data.table(weights = ..., batches = ...))
})