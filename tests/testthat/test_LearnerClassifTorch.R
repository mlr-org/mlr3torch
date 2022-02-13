test_that("LearnerTorchClassif works", {
  learner = LearnerClassifTorch$new()
  architecture = Architecture$new()
  architecture$add("linear", param_vals = list(out_features = 1))
  learner$param_set$values$criterion = nn_cross_entropy_loss
  learner$param_set$values$optimizer = optim_adam
  learner$param_set$values$architecture = architecture
  learner$param_set$values$n_epochs = 0
  learner$param_set$values$batch_size = 1
  task = tsk("wine")
  learner$train(task)
})
