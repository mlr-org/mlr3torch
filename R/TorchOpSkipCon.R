TorchOpSkipCon = R6Class("TorchOpSkipCon",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Initializes a TorchOpSkipCon
    #' @param id (`character(1)`) the id of the TorchOp.
    #' @param param_vals (`character(1)`) the parameter values
    #' @param .path Named List of length 1 containing a graph
    initialize = function(id = "skipcon", param_vals = list(), .path, .reduce = "add") {
      assert_choice(.reduce, choices = c("add"))
      assert_list(.path, len = 1L, any.missing = FALSE)
      assert_true(names(.path) != "skip")
      private$.path = .path

      param_set = ps(
        skip.bias = p_lgl(tags = "train")
      )$add(extract_paramset(.path))

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task, y) {
      x = input[["x"]]
      skip.bias = param_vals[["skip.bias"]] %??% TRUE

      # layer = self$.path$train()
      architecture = private$.path[[1]]$train(task)[[2L]]
      outputs = reduce_architecture(architecture, task, x)
      layer = outputs[["model"]]
      tensor_out = outputs[["output"]]

      out_features = tensor_out$shape[[length(tensor_out$shape)]]
      assert(length(tensor_out$shape) == length(x$shape))
      residual = nn_linear(x$shape[[length(x$shape)]], tensor_out$shape[[length(tensor_out$shape)]])

      layer = nn_parallel(residual, layer, private$.reduce)
      return(layer)

    },
    .path = NULL
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("skipcon", value = TorchOpSkipCon)

if (FALSE) {
  task = tsk("mtcars")
  graph = top("input") %>>%
    top("tokenizer", d_token = 1L) %>>%
    top("relu") %>>%
    top("linear", out_features = 10L)
  skipcon = TorchOpSkipCon$new(.path = list(a = graph))
  graph = top("input") %>>% skipcon %>>% top("model", loss = nn_mse_loss, optimizer = optim_adam)
  outputs = graph$train(task)
}
