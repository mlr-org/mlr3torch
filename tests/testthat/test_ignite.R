test_that("ignite works correctly", {
  library(mlr3pipelines)
  opt = as_torch_optimizer(ignite::optim_ignite_adamw)
  task = tsk("german_credit")
  task = po("encode")$train(list(task))[[1L]]

  neurons = rep(250, 4)

  learner1 = lrn("classif.mlp", epochs = 20, batch_size = 16, optimizer = opt,
    jit_trace = TRUE, neurons = neurons)
  learner2 = lrn("classif.mlp", epochs = 20, batch_size = 16, neurons = neurons)
  f1 = function() {
    learner1$train(task)
  }
  f2 = function() {
    learner2$train(task)
  }

  bench::mark(
    f1(),
    f2(),
    check = FALSE
  )

  p = learner$predict(task)
})
