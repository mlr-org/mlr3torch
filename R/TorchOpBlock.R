#' @title TorchOpBlock
#' @description
#' Blocks
#' @export
TorchOpBlock = R6Class("TorchOpBlock", inherit = TorchOp,
  public = list(
    initialize = function(id = "block", .graph, param_vals = list()) {
      private$.graph = assert_graph(as_graph(.graph))$clone(deep = TRUE)
      param_set = private$.graph$param_set
      assert_true("times" %nin% param_set$ids())
      tmp = ps(
        times = p_int(default = 1L, tags = c("train", "required"))
      )
      param_set$add(tmp)
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task, y) {
      times = param_vals$times
      param_vals$times = NULL
      blocks = list()

      graph_inputs = imap(inputs, function(x, nm) x[["output"]])

      for (i in seq_len(times)) {
        reduction = graphitecture_reduce(private$.graph, task, graph_inputs)
        # edges  simplify_graph(reduction$edges)
        blocks[[i]] = nn_graph$new(reduction$edges, reduction$layers)
        graph_inputs = reduction$output
      }
      invoke(nn_sequential, .args = blocks)
    },
    .graph = NULL
  )
)



#' @include mlr_torchops.R
mlr_torchops$add("block", TorchOpBlock)
