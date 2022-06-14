#' @title TorchOpBlock
#' @description
#' Blocks
#' @export
TorchOpBlock = R6Class("TorchOpBlock", inherit = TorchOp,
  public = list(
    initialize = function(id = "block", .graph, param_vals = list()) {
      private$.graph = assert_graph(as_graph(.graph))$clone(deep = TRUE)
      param_set = private$.graph$param_set
      nin = nrow(private$.graph$input)
      nout = nrow(private$.graph$output)
      assert_true(nin == 1L && nout == 1L)

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
    .build = function(inputs, param_vals, task) {
      times = param_vals$times
      param_vals$times = NULL
      blocks = list()

      g = private$.graph$clone(deep = TRUE)
      model_args = structure(
        class = "ModelArgs",
        list(
          network = nn_graph$new(),
          task = task,
          id = "__initial__",
          channel = "output",
          output = inputs$input
        )
      )


      for (i in seq_len(times - 1)) {
        gnew = private$.graph$clone(deep = TRUE)
        gnew$update_ids(prefix = sprintf("g%s", i))
        g = g %>>% gnew
      }

      network = g$train(model_args)[[1L]]$network

      return(network)
    },
    .graph = NULL
  )
)



#' @include mlr_torchops.R
mlr_torchops$add("block", TorchOpBlock)
