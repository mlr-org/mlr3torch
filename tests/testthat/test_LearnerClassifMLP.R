test_that("LearnerClassifMLP works", {
  learner = lrn("classif.mlp",
    n_layers = 2L,
    p = 0.2,
    batch_size = 16L,
    epochs = 15L,
    d_hidden = 10,
    activation = "relu",
    callbacks = cllbs("torch.progress"),
    optimizer = "adam"
  )
  result = run_autotest(learner, check_replicable = FALSE)
  expect_true(result, info = result$error)
})
