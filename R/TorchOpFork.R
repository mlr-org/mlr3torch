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
    .build = function(inputs, param_vals, task, y) {
      stop("Cannot build TorchOpFork")
    }
  )
)

mlr_torchops$add("fork", TorchOpFork)
