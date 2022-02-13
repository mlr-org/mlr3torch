test_that("LearnerTorchRegr works", {
  learner = LearnerTorchRegr$new()
  learner$param_set$values$criterion = nn_mse_loss
  learner$param_set$values$optimizer = optim_sgd
  task = tsk("mtcars")
  learner$train(task)
})
