expect_learner_torch = function(learner, check_man = TRUE, check_id = TRUE) {
  # TODO: Finish this:
  # * Test device placement of dataloader / network with "meta" device
  expect_learner(learner)
  if (check_id) expect_true(startsWith(learner$id, learner$task_type))
  expect_subset(c("loss", "optimizer", "callbacks"), formalArgs(learner$initialize))
  expect_subset(c("mlr3", "mlr3torch", "torch"), learner$packages)
}
