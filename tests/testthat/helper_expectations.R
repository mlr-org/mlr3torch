expect_learner_torch = function(learner, check_man = TRUE, check_id = TRUE) {
  expect_class(learner, "LearnerTorch")
  get("expect_learner", envir = .GlobalEnv)(learner)
  if (check_id) testthat::expect_true(startsWith(learner$id, learner$task_type))
  expect_subset(c("loss", "optimizer", "callbacks"), formalArgs(learner$initialize))
  expect_subset(c("mlr3", "mlr3torch", "torch"), learner$packages)
}
