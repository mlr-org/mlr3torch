test_that("the context exposes the callbacks of the training run", {
  seen = NULL
  spy = torch_callback("spy", on_begin = function() seen <<- self$ctx$callbacks)

  learner = lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
    callbacks = list(spy, t_clbk("history")))
  learner$train(tsk("iris"))

  expect_names(names(seen), permutation.of = c("spy", "history"))
  expect_class(seen$history, "CallbackSetHistory")
  # the live callbacks, not copies: they share the context of the run
  expect_identical(seen$history$ctx, seen$spy$ctx)
})

test_that("the epoch is 0 during the 'begin' stage", {
  epoch = NULL
  spy = torch_callback("spy", on_begin = function() epoch <<- self$ctx$epoch)

  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10, callbacks = spy)
  learner$train(tsk("iris"))

  expect_equal(epoch, 0L)
})
