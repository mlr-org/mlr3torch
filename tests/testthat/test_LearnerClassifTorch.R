test_that("LearnerTorchClassif works", {
  task = tsk("iris")
  learner = LearnerClassifTorch$new()
  architecture = Architecture$new()
  architecture$add("linear", param_vals = list(out_features = 3))
  architecture$add("softmax", param_vals = list(dim = 2L))
  learner$param_set$values$criterion = nn_cross_entropy_loss
  learner$param_set$values$optimizer = optim_adam
  learner$param_set$values$architecture = architecture
  learner$param_set$values$n_epochs = 1
  learner$param_set$values$batch_size = 2
  learner$train(task)
})
