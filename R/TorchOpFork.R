#' @title
TorchOpFork = R6Class("TorchOpFork",
  inherit = TorchOp,
  public = list(
    #' @param names (`character()`) The names of the branches that are create from this fork.
    names = NULL,
    initialize = function(id = "fork", .outnum, param_vals = list()) {
      assert_count(.outnum)
      param_set = ps()
      output = data.table(
        name = paste0("output", seq_len(.outnum)),
        train = rep("*", times = .outnum),
        predict = rep("*", times = .outnum)
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        output = output
      )
    }
  ),
  private = list(
    .build = function(input, param_vals, task, y) {
      stop("Cannot build layer for object of class TorchOpFork")

    },
    .train = function(inputs) {
      # TODO: don't repeat yourself (this is copied from TorchOp)
      if (!is.null(self$state)) { # this means the architecture is already built
        return(inputs)
      }
      if (test_r6(inputs[[1L]], "Task")) { # this means this torchop is the first in the architecture
        # and we have to build the architecture
        task = inputs[[1L]]
        architecture = Architecture$new()
      } else {
        task = inputs[["input"]][["task"]]
        architecture = inputs[["input"]][["architecture"]]
      }
      architecture = inputs[["architecture"]]
      outputs = map(
        self$outnum,
        function(i) {
          architecture = architecture$copy(deep = FALSE)
          architecture$ptr = paste0(self$id, i)
          list(task = task, architecture = architecture)
        }
      )
      return(outpus)
    }
  )
)

mlr_torchops$add("fork", TorchOpFork)


if (FALSE) {

  task = tsk("mtcars")
  graph = top("linear", id = "linear0") %>>%
    top("fork", .outnum = 2L) %>>%
    gunion(
      list(
        a = top("linear", out_features = 10) %>>% top("relu"),
        b = top("linear", out_features = 10)
      )
    ) %>>%
    top("merge", .innum = 2L, method = "add") %>>%
    top("linear", out_features = 1)

  # top("branch", branches = c("a", "b"))
  graph$train(task)
}
