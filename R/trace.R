nn_trace = nn_module("nn_trace",
  initialize = function(module, inputs) {
    if (!is.list(inputs)) {
      inputs = list(inputs)
    }
    was_training = module$training
    on.exit(module$train(was_training))
    module$train()
    self$module = mlr3misc::invoke(jit_trace, func = module, .args = inputs)
    module$eval()
    forward_eval = mlr3misc::invoke(jit_trace, func = module, .args = inputs)
    attr(forward_eval, "module") <- NULL
    class(forward_eval) <- NULL
    self$forward_eval = forward_eval
  },
  forward = function(...) {
    if (self$training) {
      self$module(...)
    } else {
      self$forward_eval(...)
    }
  }
)

