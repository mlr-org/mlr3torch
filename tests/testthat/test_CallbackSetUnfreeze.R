check_frozen = torch_callback("check_frozen",
  initialize = function(starting_weights, unfreeze) {
    self$starting_weights = starting_weights
    # consider supporting character vectors
    self$unfreeze = unfreeze
  },
  on_epoch_end = function() {
    if ("epoch" %in% names(self$unfreeze)) {
      if (self$ctx$epoch %in% self$unfreeze$epoch) {
        weights = (self$unfreeze[epoch == self$ctx$epoch]$unfreeze)[[1]](names(self$ctx$network$parameters))
        walk(self$ctx$network$parameters[weights], function(param) print(param$requires_grad))
      }
    }
  },
  on_batch_end = function() {
    if ("batch" %in% names(self$unfreeze)) {
      batch_num = (self$ctx$epoch - 1) * length(self$ctx$loader_train) + self$ctx$step
      if (batch_num %in% self$unfreeze$batch) {
        weights = (self$unfreeze[batch == batch_num]$unfreeze)[[1]](names(self$ctx$network$parameters))
        walk(self$ctx$network$parameters[weights], function(param) print(param$requires_grad))
      }
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

# test_that("unfreezing on epochs works in the end", {
#   n_epochs = 10
#
#   task = tsk("iris")
#
#   mlp = lrn("classif.mlp",
#             callbacks = t_clbk("unfreeze"),
#             epochs = 10, batch_size = 150, neurons = c(100, 200, 300)
#   )
#
#   mlp$param_set$set_values(cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight"))))
#
#   mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
#       epoch = 2,
#       unfreeze = select_name("0.weight")
#     )
#   )
#
#   mlp$train(task)
#
#   expect_true(mlp$network$parameters[[select_name("0.weight")(names(mlp$network$parameters))]]$requires_grad)
#   expect_false(mlp$network$parameters[[select_name("3.weight")(names(mlp$network$parameters))]]$requires_grad)
# })
#
# test_that("unfreezing on batches works in the end", {
#   n_epochs = 10
#
#   task = tsk("iris")
#
#   mlp = lrn("classif.mlp",
#             callbacks = t_clbk("unfreeze"),
#             epochs = 10, batch_size = 50, neurons = c(100, 200, 300)
#   )
#
#   mlp$param_set$set_values(cb.unfreeze.starting_weights = select_invert(select_name(c("0.weight", "3.weight"))))
#
#   mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
#       batch = 2,
#       unfreeze = select_name("0.weight")
#     )
#   )
#
#   mlp$train(task)
#
#   expect_true(mlp$network$parameters[[select_name("0.weight")(names(mlp$network$parameters))]]$requires_grad)
#   expect_false(mlp$network$parameters[[select_name("3.weight")(names(mlp$network$parameters))]]$requires_grad)
# })

test_that("freezing with epochs works at the correct time", {
  n_epochs = 10

  task = tsk("iris")

  mlp = lrn("classif.mlp",
            callbacks = list(t_clbk("unfreeze"), check_frozen),
            epochs = 10, batch_size = 50, neurons = c(100, 200, 300)
  )

  frozen_weights_at_start = c("0.weight", "0.bias", "3.bias", "3.weight")

  mlp$param_set$set_values(cb.unfreeze.starting_weights = select_invert(select_name(frozen_weights_at_start)))
  mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
      epoch = seq_along(frozen_weights_at_start),
      unfreeze = map(frozen_weights_at_start, function(name) select_name(name))
    )
  )

  mlp$param_set$set_values(cb.check_frozen.starting_weights = select_invert(select_name(frozen_weights_at_start)))
  mlp$param_set$set_values(cb.check_frozen.unfreeze = data.table(
      epoch = seq_along(frozen_weights_at_start),
      unfreeze = map(frozen_weights_at_start, function(name) select_name(name))
    )
  )

  train_output = capture_output(mlp$train(task))
  expect_match(train_output, "TRUE")
  expect_no_match(train_output, "FALSE")
})

test_that("freezing with batches works at the correct time", {
  n_epochs = 10

  task = tsk("iris")

  mlp = lrn("classif.mlp",
            callbacks = list(t_clbk("unfreeze"), check_frozen),
            epochs = 10, batch_size = 50, neurons = c(100, 200, 300)
  )

  frozen_weights_at_start = c("0.weight", "0.bias", "3.bias", "3.weight")

  mlp$param_set$set_values(cb.unfreeze.starting_weights = select_invert(select_name(frozen_weights_at_start)))
  mlp$param_set$set_values(cb.unfreeze.unfreeze = data.table(
    batch = seq_along(frozen_weights_at_start),
    unfreeze = map(frozen_weights_at_start, function(name) select_name(name))
  )
  )

  mlp$param_set$set_values(cb.check_frozen.starting_weights = select_invert(select_name(frozen_weights_at_start)))
  mlp$param_set$set_values(cb.check_frozen.unfreeze = data.table(
    batch = seq_along(frozen_weights_at_start),
    unfreeze = map(frozen_weights_at_start, function(name) select_name(name))
  )
  )

  train_output = capture_output(mlp$train(task))
  expect_match(train_output, "TRUE")
  expect_no_match(train_output, "FALSE")
})
