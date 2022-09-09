test_that("LearnerClassifMLP works", {
  learner = lrn("classif.mlp",
    layers = 2L,
    p = 0.2,
    batch_size = 16L,
    epochs = 15L,
    d_hidden = 10,
    activation = "relu",
    optimizer = "adam"
  )
  result = run_autotest(learner, check_replicable = FALSE)
  expect_true(result, info = result$error)
})
if (FALSE) {
  task = tsk("iris")
  learner$param_set$values$epochs = 1
  learner$param_set$values$opt.lr = 0.3
  learner$train(task)
  learner$disassemble()
  debug(learner$assemble)
  learner$assemble()
  p = learner$predict(task)
}
