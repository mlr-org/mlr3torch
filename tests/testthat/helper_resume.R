# Helpers for the tests that resume a training run from a checkpoint.
make_checkpoint = function(epochs = 2L, freq = 1L, path = tempfile(), callbacks = list(),
  task = tsk("iris"), ...) {
  learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
    callbacks = c(list(t_clbk("checkpoint", freq = freq, path = path)), callbacks), ...)
  learner$train(task)
  learner
}

task_with_valid = function(ids = 1:30) {
  task = tsk("iris")
  task$internal_valid_task = ids
  task
}

resumer = function(epochs, path, callbacks = list(), ...) {
  learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
    callbacks = callbacks, ...)
  learner$param_set$set_values(resume = path)
  learner
}

crashing_run = function(path, epochs, fail_at, callback, values = list(), task = tsk("iris"), ...) {
  crash = torch_callback("crash",
    on_epoch_begin = function() if (self$ctx$epoch == fail_at) stop("crash"))
  learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10, seed = 1,
    callbacks = list(t_clbk("checkpoint", freq = 1, path = path), callback, crash), ...)
  learner$param_set$set_values(.values = values)
  expect_error(learner$train(task), "crash")
  learner
}
