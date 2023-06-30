expect_learner_torch = function(learner) {
  expect_subset(c("loss", "optimizer", "callbacks"), formalArgs(learner$initialize))
  expect_class(learner, "LearnerTorch")
  testthat::expect_true(grepl("^LearnerTorch", class(learner)[[1L]], perl = TRUE))
}
