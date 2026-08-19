test_that("autotest", {
  cb = t_clbk("progress")
  expect_torch_callback(cb)
})

test_that("the training time is carried across a resume", {
  path = tempfile()
  first = lrn("classif.mlp", epochs = 2, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 1, path = path), t_clbk("progress")))
  suppressMessages(capture.output(first$train(tsk("iris"))))

  elapsed = first$model$callbacks$progress$elapsed
  expect_number(elapsed, lower = 0)

  # the checkpoint of epoch 1 was written before the run finished, so it holds less time than the
  # completed run does
  expect_lte(readRDS(file.path(path, "state1.rds"))$callbacks$progress$elapsed, elapsed)

  resumed = lrn("classif.mlp", epochs = 4, batch_size = 50, neurons = 10,
    callbacks = t_clbk("progress"))
  resumed$param_set$set_values(resume = path)
  stdout = suppressMessages(capture.output(resumed$train(tsk("iris"))))

  # the total covers both runs, so it is at least what the first one alone took
  expect_gte(resumed$model$callbacks$progress$elapsed, elapsed)
  expect_match(stdout[length(stdout)], "Finished training for 4 epochs .*s total")
})

test_that("manual test", {
  learner = lrn("classif.mlp", epochs = 1, batch_size = 1,
    measures_train = msr("classif.acc"), measures_valid = msr("classif.ce"), callbacks = t_clbk("progress"),
    drop_last = FALSE, shuffle = TRUE, validate = "predefined"
  )
  task = tsk("iris")
  task$internal_valid_task = task$clone(deep = TRUE)$filter(2)
  task$filter(1)

  # Because the validation is so short, it does not show in the example
  # We can make it longer by adding some sleep through callbacks
  # Still, this is not captured by capture.output(), so one has to manually inspect that it works
  # callbacks = list(t_clbk("progress"), cbutil)
  # cbutil = torch_callback("util", on_batch_valid_begin = function() Sys.sleep(1))

  stdout = suppressMessages(capture.output(learner$train(task)))

  expected = c(
    "Epoch 1 started",
    "Validation for epoch 1 started",
    "",
    "[Summary epoch 1]",
    "------------------",
    "Measures (Train):",
    " * classif.acc =",
    "Measures (Valid):",
    " * classif.ce =",
    "",
    "Finished training for 1 epochs"
  )

  expect_true(length(stdout) == length(expected))
  expect_true(all(map_lgl(seq_along(stdout), function(i) startsWith(stdout[[i]], expected[[i]]))))

  # does not throw with different eval_freq
  learner$param_set$set_values(eval_freq = 2)
  expect_error(capture.output(learner$train(task)), regexp = NA)
})

describe("resuming", {
  it("reports the time before this run and the time it took itself", {
    path = tempfile()
    make_checkpoint(epochs = 2L, path = path, callbacks = list(t_clbk("progress")))

    resumed = resumer(4L, path, callbacks = t_clbk("progress"))
    out = capture.output(resumed$train(tsk("iris")))
    finished = out[grepl("^Finished training", out)]
    expect_match(finished, "s total: .*s before this run, .*s in it")
  })


  it("the progress callback does not interfere", {
    # it keeps no state, so the point is that resuming a run that has one works at all
    path = tempfile()
    make_checkpoint(epochs = 2L, path = path, callbacks = list(t_clbk("progress")))

    # the callback prints via catn(), so the output is captured rather than suppressed
    resumed = resumer(4L, path, callbacks = t_clbk("progress"))
    expect_no_warning(capture.output(resumed$train(tsk("iris"))))
    expect_equal(resumed$model$epochs, 4L)
  })
})
