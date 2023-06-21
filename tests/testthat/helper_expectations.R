expect_learner_torch = function(learner, check_man = TRUE) {
  # TODO: Finish this:
  # * Test device placement of dataloader / network with "meta" device
  expect_learner(learner)
  expect_subset(c("loss", "optimizer", "callbacks"), formalArgs(learner$initialize))
  testthat::expect_true(grepl("^Learner(Classif|Regr)Torch", class(learner)[[1L]], perl = TRUE))
  expect_true(inherits(learner, "LearnerClassifTorch") || inherits(learner, "LearnerRegrTorch"))
  expect_subset(c("mlr3", "mlr3torch", "torch"), learner$packages)
}
