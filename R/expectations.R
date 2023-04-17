expect_learner_torch = function(learner) {
  expect_subset(c("loss", "optimizer", "callbacks"), formalArgs(learner$initialize))
  expect_true(grepl("^Learner(Classif|Regr)Torch", class(learner)[[1L]], perl = TRUE))
}
