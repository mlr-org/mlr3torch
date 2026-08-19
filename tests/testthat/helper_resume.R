# Helpers for the tests that resume a training run from a checkpoint.

# i.e. the `path` parameter of LearnerTorch, continuing what t_clbk("checkpoint") wrote
make_checkpoint = function(epochs = 2L, freq = 1L, path = tempfile(), callbacks = list(), ...) {
  learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
    callbacks = c(list(t_clbk("checkpoint", freq = freq, path = path)), callbacks), ...)
  learner$train(tsk("iris"))
  learner
}

resumer = function(epochs, path, callbacks = list(), ...) {
  learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
    callbacks = callbacks, ...)
  learner$param_set$set_values(resume = path)
  learner
}

# Checkpoints into `path` and then fails at the beginning of epoch `fail_at`, so that the epochs
# before it are written and there is something to resume from. `values` are learner parameter
# values, `...` is passed to lrn().
crashing_run = function(path, epochs, fail_at, callback, values = list(), ...) {
  crash = torch_callback("crash",
    on_epoch_begin = function() if (self$ctx$epoch == fail_at) stop("crash"))
  learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10, seed = 1,
    callbacks = list(t_clbk("checkpoint", freq = 1, path = path), callback, crash), ...)
  learner$param_set$set_values(.values = values)
  expect_error(learner$train(tsk("iris")), "crash")
  learner
}
