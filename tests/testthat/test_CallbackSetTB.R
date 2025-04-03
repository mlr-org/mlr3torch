skip_if_not_installed("tfevents")

event_tag_is = function(event, tag_name) {
  if (is.null(event)) FALSE else event["tag"] == tag_name
}

test_that("autotest", {
  cb = t_clbk("tb", path = tempfile())
  expect_torch_callback(cb, check_man = TRUE)
})

test_that("metrics are logged correctly", {
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

test_that("throws an error when using existing directory", {
  path = tempfile()
  dir.create(path)
  cb = t_clbk("tb", path = path, log_train_loss = TRUE)
  expect_error(cb$generate(), "already exists")
})
