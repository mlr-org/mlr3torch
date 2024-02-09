expect_learner_torch = function(learner, check_man = TRUE, check_id = TRUE) {
  # TODO: Finish this:
  # * Test device placement of dataloader / network with "meta" device

  # because expect_learner_torch lives in the mlr3torch namespace, but expect_learner lives in
  # the mlr3 namespace, we need to use get() to access the function
  get("expect_learner", envir = .GlobalEnv)(learner)
  if (check_id) expect_true(startsWith(learner$id, learner$task_type))
  expect_subset(c("loss", "optimizer", "callbacks"), formalArgs(learner$initialize))
  expect_subset(c("mlr3", "mlr3torch", "torch"), learner$packages)
}
