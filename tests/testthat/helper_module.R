testmodule_linear = nn_module(
  initialize = function(task) {
    out = get_nout(task)
    self$linear = nn_linear(length(task$feature_names), out)
  },
  forward = function(x) {
    self$linear(x)
  }
)
