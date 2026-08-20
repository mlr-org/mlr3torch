skip_if_not_installed("tfevents")

event_tag_is = function(event, tag_name) {
  if (is.null(event)) FALSE else event["tag"] == tag_name
}

test_that("autotest", {
  cb = t_clbk("tb", path = tempfile())
  expect_torch_callback(cb, check_man = TRUE)
})

test_that("metrics are logged correctly", {
  skip_if_not_installed("tibble")
  cb = t_clbk("tb")

  task = tsk("iris")

  n_epochs = 10

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = 150, neurons = 10,
            validate = 0.2,
            measures_valid = msrs(c("classif.acc", "classif.ce")),
            measures_train = msrs(c("classif.acc", "classif.ce"))
  )
  mlp$param_set$set_values(cb.tb.path = tempfile())
  mlp$param_set$set_values(cb.tb.log_train_loss = TRUE)

  mlp$train(task)

  events = mlr3misc::map(tfevents::collect_events(mlp$param_set$get_values()$cb.tb.path)$summary, unlist)

  n_train_loss_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "train.loss"))
  n_train_acc_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "train.classif.acc"))
  n_train_ce_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "train.classif.ce"))
  n_valid_acc_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "valid.classif.acc"))
  n_valid_ce_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "valid.classif.ce"))

  expect_equal(n_train_loss_events, n_epochs)
  expect_equal(n_train_acc_events, n_epochs)
  expect_equal(n_train_ce_events, n_epochs)
  expect_equal(n_valid_acc_events, n_epochs)
  expect_equal(n_valid_ce_events, n_epochs)
})

test_that("eval_freq works", {
  skip_if_not_installed("tibble")
  task = tsk("iris")

  n_epochs = 9
  eval_freq = 4

  mlp = lrn("classif.mlp",
            callbacks = t_clbk("tb"),
            epochs = n_epochs, batch_size = 150, neurons = 200,
            validate = 0.2,
            measures_valid = msrs(c("classif.acc", "classif.ce")),
            measures_train = msrs(c("classif.acc", "classif.ce")),
            eval_freq = eval_freq
  )
  mlp$param_set$set_values(cb.tb.path = tempfile())
  mlp$param_set$set_values(cb.tb.log_train_loss = TRUE)

  mlp$train(task)

  events = mlr3misc::map(tfevents::collect_events(mlp$param_set$get_values()$cb.tb.path)$summary, unlist)

  n_train_loss_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "train.loss"))
  n_train_acc_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "train.classif.acc"))
  n_train_ce_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "train.classif.ce"))
  n_valid_acc_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "valid.classif.acc"))
  n_valid_ce_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "valid.classif.ce"))

  expect_equal(n_train_loss_events, n_epochs)
  expect_equal(n_train_acc_events, ceiling(n_epochs / eval_freq))
  expect_equal(n_train_ce_events, ceiling(n_epochs / eval_freq))
  expect_equal(n_valid_acc_events, ceiling(n_epochs / eval_freq))
  expect_equal(n_valid_ce_events, ceiling(n_epochs / eval_freq))
})

test_that("we can disable training loss tracking", {
  skip_if_not_installed("tibble")
  task = tsk("iris")

  mlp = lrn("classif.mlp",
            callbacks = t_clbk("tb"),
            epochs = 10, batch_size = 150, neurons = 200
  )
  mlp$param_set$set_values(cb.tb.path = tempfile())
  mlp$param_set$set_values(cb.tb.log_train_loss = FALSE)

  mlp$train(task)

  events = mlr3misc::map(tfevents::collect_events(mlp$param_set$get_values()$cb.tb.path)$summary, unlist)

  n_train_loss_events = sum(mlr3misc::map_lgl(events, event_tag_is, tag_name = "train.loss"))

  expect_equal(n_train_loss_events, 0)
})

test_that("an existing directory is accepted only when it holds events", {
  # a resumed run logs into the folder its predecessor wrote, so that one is allowed
  path = tempfile()
  dir.create(path)
  expect_no_error(t_clbk("tb", path = path, log_train_loss = TRUE)$generate())

  file.create(file.path(path, "events.out.tfevents.1234567890.host"))
  expect_no_error(t_clbk("tb", path = path, log_train_loss = TRUE)$generate())

  # a folder holding anything else is not ours to write into
  unrelated = tempfile()
  dir.create(unrelated)
  writeLines("not an event file", file.path(unrelated, "notes.txt"))
  expect_error(t_clbk("tb", path = unrelated, log_train_loss = TRUE)$generate(), "already exists")
})

describe("resuming", {
  it("a resumed run continues the training loss at the step the previous one stopped at", {
    skip_if_not_installed("tibble")
    task = tsk("iris")
    checkpoint_path = tempfile()
    first_path = tempfile()

    first = lrn("classif.mlp", epochs = 2, batch_size = 150, neurons = 10,
      callbacks = list(t_clbk("checkpoint", freq = 1, path = checkpoint_path), t_clbk("tb")))
    first$param_set$set_values(cb.tb.path = first_path, cb.tb.log_train_loss = TRUE)
    first$train(task)

    loss_steps = function(path) {
      events = tfevents::collect_events(path)
      events$step[map_lgl(map(events$summary, unlist), event_tag_is, tag_name = "train.loss")]
    }
    # one batch per epoch, so the two epochs were logged under the run's batch steps 1 and 2
    expect_equal(loss_steps(first_path), c(1, 2))

    # `path` must not exist yet, so a resumed run logs into a folder of its own -- what carries over
    # is the step, which comes from `ctx$global_step`, so that the curve continues instead of
    # starting over
    resumed_path = tempfile()
    resumed = lrn("classif.mlp", epochs = 4, batch_size = 150, neurons = 10, callbacks = t_clbk("tb"))
    resumed$param_set$set_values(cb.tb.path = resumed_path, cb.tb.log_train_loss = TRUE,
      resume = checkpoint_path)
    resumed$train(task)

    expect_equal(loss_steps(resumed_path), c(3, 4))
    # the callback has no state of its own, the step comes from the context
    expect_null(resumed$model$callbacks$tb)
  })

  it("the tensorboard callback logs into the folder of the run it continues", {
    skip_if_not_installed("tfevents")
    path = tempfile()
    logdir = tempfile()
    tb = t_clbk("tb", path = logdir, log_train_loss = TRUE)
    make_checkpoint(epochs = 2L, path = path, callbacks = list(tb))

    # the same `logdir`, which the first run created -- both halves belong to one TensorBoard run
    resumed = resumer(4L, path, callbacks = tb)
    expect_no_warning(resumed$train(tsk("iris")))
    expect_equal(resumed$model$epochs, 4L)

    events = tfevents::collect_events(logdir)
    losses = events$step[map_lgl(map(events$summary, unlist), event_tag_is, tag_name = "train.loss")]
    # 3 batches per epoch across the two runs, counted through rather than restarted
    expect_equal(losses, 1:12)
  })
})
