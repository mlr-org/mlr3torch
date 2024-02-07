test_that("autotest", {
  cb = t_clbk("progress")
  expect_torch_callback(cb)
})

test_that("manual test", {
  learner = lrn("classif.mlp", epochs = 1, batch_size = 1,
    measures_train = msr("classif.acc"), measures_valid = msr("classif.ce"), callbacks = t_clbk("progress"),
    drop_last = FALSE, shuffle = TRUE
  )
  task = tsk("iris")
  task$row_roles = list(use = 1, test = 2, holdout = integer(0))

  # Because the validation is so short, it does not show in the example
  # We can make it longer by adding some sleep through callbacks
  # Still, this is not captured by capture.output(), so one has to manually inspect that it works
  # callbacks = list(t_clbk("progress"), cbutil)
  # cbutil = torch_callback("util", on_batch_valid_begin = function() Sys.sleep(1))

  stdout = suppressMessages(capture.output(learner$train(task)))

  expected = c(
    "Epoch 1",
    "",
    "[Summary epoch 1]",
    "------------------",
    "Measures (Train):",
    " * classif.acc =",
    "Measures (Valid):",
    " * classif.ce ="
  )

  expect_true(length(stdout) == length(expected))
  expect_true(all(map_lgl(seq_along(stdout), function(i) startsWith(stdout[[i]], expected[[i]]))))
})
