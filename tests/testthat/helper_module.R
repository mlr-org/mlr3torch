testmodule_linear = nn_module(
  initialize = function(task) {
    out = output_dim_for(task)
    self$linear = nn_linear(length(task$feature_names), out)
  },
  forward = function(x) {
    self$linear(x)
  }
)
