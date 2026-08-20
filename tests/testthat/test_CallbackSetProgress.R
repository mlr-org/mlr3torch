test_that("autotest", {
  cb = t_clbk("progress")
  expect_torch_callback(cb)
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
    "Epoch 1/1 started",
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
  it("continues the run the checkpoint came from", {
    path = tempfile()
    first = make_checkpoint(epochs = 2L, path = path, callbacks = list(t_clbk("progress")))

    elapsed = first$model$callbacks$progress$elapsed
    expect_number(elapsed, lower = 0)
    # the checkpoint of epoch 1 was written before the run finished, so it holds less time than the
    # completed run does
    expect_lte(readRDS(file.path(path, "state1.rds"))$callbacks$progress$elapsed, elapsed)

    # the callback keeps no state of its own, so resuming a run that has one has to work at all
    resumed = resumer(4L, path, callbacks = t_clbk("progress"))
    # the callback prints via catn(), so the output is captured rather than suppressed
    out = capture.output(expect_no_warning(resumed$train(tsk("iris"))))
    expect_equal(resumed$model$epochs, 4L)

    # the total covers both runs, so it is at least what the first one alone took
    expect_gte(resumed$model$callbacks$progress$elapsed, elapsed)
    expect_match(out[grepl("^Finished training", out)],
      "^Finished training for 4 epochs .*s total: .*s before this run, .*s in it")
    # only the epochs this run trains itself, numbered as what they are
    expect_match(out[grepl("^Epoch ", out)], "^Epoch [34]/4 started")
  })
})
