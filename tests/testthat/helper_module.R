testmodule_linear = nn_module(
  initialize = function(task) {
    out = if (task$task_type == "classif") length(task$class_names) else 1
    self$linear = nn_linear(length(task$feature_names), out)
  },
  forward = function(x) {
    self$linear(x)
  }
)
