#' @title Repeat a Segment of a Neural Network
#' @description
#' Repeats a Segment of a Neural Network usually referred to as a block.
#' @param block (`Graph` or `PipeOp`)\cr Block that is repeated.
#' @template param_id
#' @template param_param_vals
#'
#' @export
TorchOpRepeat = R6Class("TorchOpRepeat", inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(block, id = "rep", param_vals = list()) {
      private$.graph = assert_graph(as_graph(block))$clone(deep = TRUE)
      param_set = private$.graph$param_set
      assert_true(nrow(private$.graph$input) == 1 && nrow(private$.graph$output) == 1)

      assert_true("times" %nin% param_set$ids())
      tmp = ps(
        times = p_int(default = 1L, tags = c("train", "required"))
      )
      tmp$values$times = 1L
      param_set$add(tmp)
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      param_vals = self$param_set$get_values(tag = "train")
      times = param_vals$times
      param_vals$times = NULL
      blocks = list()

      # TODO: This should be integrated into the edges of the actual network
      g = private$.graph$clone(deep = TRUE)
      g$update_ids(prefix = "b1.")
      model_args = structure(
        class = "ModelArgs",
        list(
          network = nn_graph(),
          task = task,
          id = "__initial__",
          channel = "output",
          output = inputs$input
        )
      )

      for (i in (1 + seq_len(times - 1))) {
        gnew = private$.graph$clone(deep = TRUE)
        gnew$update_ids(prefix = sprintf("b%s.", i))
        g = g %>>% gnew
      }

      network = g$train(model_args)[[1L]]$network

      return(network)
    },
    .graph = NULL
  )
)



#' @include mlr_torchops.R
mlr_torchops$add("repeat", TorchOpRepeat)
