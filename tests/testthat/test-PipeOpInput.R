test_that("Works", {
  task = tsk("pima")
  pipeop = PipeOpInput$new()
  output = pipeop$train(list(task))
  expect_list(output)
  expect_true(identical(task, output[["task"]]))
  expect_r6(output[["architecture"]], "Architecture")
})
