# Tests that the learner works

autotest_classif = function(l) {
  task = toytask()
  l = lrn("classif.alexnet")
  task$row_roles$use = 10L
  l$param_set$values$device = "meta"
  expect_error(l$train(task), regexp = NA)

  env = environment()

  default_valid_fn
  tracer = function() {
    # assign("training", parent.env()$ngtwork$training, envir = env)
    if (parent.frame()$network$training) {
      stopf("")
    }
    print(parent.frame()$network$training)
    print("hallo")
  }

  l = lrn("classif.alexnet")
  l$param_set$values$device = "cpu"
  l$param_set$values$batch_size = 1L
  l$param_set$values$epochs = 1L

  trace(default_train_fn, tracer)
  l$param_set$values$train_fn = default_train_fn


  l$train(task)

  pred1 = l$predict(task)
  l$train(task)
}
