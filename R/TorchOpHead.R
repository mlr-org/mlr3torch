TorchOpHead = R6Class("TorchOpHead",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "head", param_vals = list()) {
      param_set = ps(
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    },
    names_in = c("batch", "feature", "token"),
    names_out = function(names_in) {
      c("batch", "response")
    }
  ),
  private = list(
    .operator = "head",
    .build = function(x, param_vals, task) {
      bias = param_vals[["bias"]] %??% TRUE
      in_features = x$shape[[2L]]

      if (task$task_type == "classif") {
        out_features = length(task$levels(task$col_roles$target))
      } else if (task$task_type == "regr") {
        out_features = 1L
      } else {
        stop("Task type not supported!")
      }

      layer = nn_linear(in_features, out_features)

      return(layer)
    },
    .name = function(x) {
      x
    }
  )
)

mlr_torchops$add("head", TorchOpHead)
# TorchOpHeadCLS = R6Class("TorchOpHeadCLS",
#   inherit = TorchOp,
#   public = list(
#     initialize = function(id = "head", param_vals = list()) {
#       param_set = ps(
#         cls = p_lgl(tags = c("train")),
#         pos =
#       )
#       super$initialize(
#         id = id,
#         param_set = param_set,
#         param_vals = param_vals
#       ),
#     }
#     names_in = list(
#
#     )
#   ),
#   private = list(
#     .operator = "head",
#     .build = function(input, param_vals, task) {
#
#     },
#     .name = function(x) {
#       x
#     }
#   )
# )
#
