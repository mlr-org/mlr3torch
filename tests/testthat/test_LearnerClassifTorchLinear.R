test_that("LearnerClassifTorchLinear works", {
  learner = lrn("classif.torch_linear", batch_size = 1, epochs = 10)
  result = run_autotest(learner)
  expect_true(result, info = result$info)
})
