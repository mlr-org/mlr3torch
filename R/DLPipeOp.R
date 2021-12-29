#DLPipeOp = R6Class("DLPipeOp",
#  inherit = mlr3pipelines::PipeOp,
#  public = list(
#    build = function(input) {
#      stop
#    },
#  ),
#  private = list(
#    .train = function(inputs) {
#      task = inputs[["task"]]
#      architecture = inputs[["architecture"]]
#      self$state = "trained"
#      if (is.null(inputs[["architecture"]])) {
#        architecture = Architecture$new()
#      } else {
#        architecture = architecture$add(
#          list("")
#        )
#      }
#      architecture =
#        output = list(task = task, architecture = architecture)
#      return(output)
#    },
#    .predict = function(inputs) {
#      task = inputs[["input"]]
#      output = list(input = task, architecture = NULL)
#    },
#    .build = function() {
#      stop("Abstract base class.")
#    }
#  )
#)
