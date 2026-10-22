test_that("the context exposes the callbacks of the training run", {
  state = new.env()
  spy = torch_callback("spy", on_begin = function() {
    state$seen = self$ctx$callbacks
  })

  learner = lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
    callbacks = list(spy, t_clbk("history")))
  learner$train(tsk("iris"))

  seen = state$seen
  expect_names(names(seen), permutation.of = c("spy", "history"))
  expect_class(seen$history, "CallbackSetHistory")
  # the live callbacks, not copies: they share the context of the run
  expect_identical(seen$history$ctx, seen$spy$ctx)
})

test_that("the epoch is 0 during the 'begin' stage", {
  state = new.env()
  spy = torch_callback("spy", on_begin = function() {
    state$epoch = self$ctx$epoch
  })

  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10, callbacks = spy)
  learner$train(tsk("iris"))

  expect_equal(state$epoch, 0L)
})

test_that("last_loss is the numeric loss of the last batch", {
  state = new.env()
  state$losses = list()
  state$expected = numeric()
  spy = torch_callback("spy", on_batch_end = function() {
    state$losses[[length(state$losses) + 1L]] = self$ctx$last_loss
    state$expected = c(state$expected, self$ctx$loss_fn(self$ctx$y_hats, self$ctx$batch$y)$item())
  })

  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10, callbacks = spy)
  learner$train(tsk("iris"))

  losses = state$losses
  expected = state$expected
  # iris has 150 observations, so there are 3 batches per epoch
  expect_length(losses, 6L)
  for (loss in losses) expect_number(loss)
  expect_equal(unlist(losses), expected, tolerance = 1e-6)
})
