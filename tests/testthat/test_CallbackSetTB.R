library(tfevents)

test_that("basic", {
    cb = t_clbk("tb", path = tempfile())
    expect_torch_callback(cb, check_man = FALSE)
})

# TODO: investigate what's happening when there is only a single epoch (why don't we log anything?)
test_that("a simple example works", {
    # using a temp dir
    cb = t_clbk("tb")

    task = tsk("iris")
    n_epochs = 10
    batch_size = 50
    neurons = 200

    pth0 = tempfile()

    mlp = lrn("classif.mlp",
          callbacks = cb,
          epochs = n_epochs, batch_size = batch_size, neurons = neurons,
          validate = 0.2,
          measures_valid = msrs(c("classif.acc", "classif.ce")),
          measures_train = msrs(c("classif.acc", "classif.ce"))
    )
    mlp$param_set$set_values(cb.tb.path = pth0)

    mlp$train(task)

    events = collect_events(pth0)$summary %>%
        mlr3misc::map(unlist)

    n_last_loss_events = mlr3misc::map(events, \(x) x["tag"] == "train.loss") %>%
        unlist() %>%
        sum()

    n_valid_acc_events = mlr3misc::map(events, \(x) x["tag"] == "valid.classif.acc") %>%
      unlist() %>%
      sum()

    n_valid_ce_events = mlr3misc::map(events, \(x) x["tag"] == "valid.classif.ce") %>%
      unlist() %>%
      sum()

    # TODO: refactor to expect a specific ordering of the events list
    expect_equal(n_last_loss_events, n_epochs)
    expect_equal(n_valid_acc_events, n_epochs)
    expect_equal(n_valid_ce_events, n_epochs)
})

test_that("eval_freq works", {
  # using a temp dir
  cb = t_clbk("tb")

  task = tsk("iris")
  n_epochs = 9
  batch_size = 50
  neurons = 200
  eval_freq = 4

  pth0 = tempfile()

  mlp = lrn("classif.mlp",
            callbacks = cb,
            epochs = n_epochs, batch_size = batch_size, neurons = neurons,
            validate = 0.2,
            measures_valid = msrs(c("classif.acc", "classif.ce")),
            measures_train = msrs(c("classif.acc", "classif.ce")),
            eval_freq = eval_freq
  )
  mlp$param_set$set_values(cb.tb.path = pth0)

  mlp$train(task)

  events = collect_events(pth0)$summary %>%
    mlr3misc::map(unlist)

  n_last_loss_events = mlr3misc::map(events, \(x) x["tag"] == "train.loss") %>%
    unlist() %>%
    sum()

  n_valid_acc_events = mlr3misc::map(events, \(x) x["tag"] == "valid.classif.acc") %>%
    unlist() %>%
    sum()

  n_valid_ce_events = mlr3misc::map(events, \(x) x["tag"] == "valid.classif.ce") %>%
    unlist() %>%
    sum()

  expect_equal(n_last_loss_events, n_epochs)
  expect_equal(n_valid_acc_events, ceiling(n_epochs / 4))
  expect_equal(n_valid_ce_events, ceiling(n_epochs / 4))
})

test_that("throws an error when using existing directory", {
  path = tempfile()
  dir.create(path)
  cb = t_clbk("tb", path = path)
  expect_error(cb$generate(), "already exists")
})
