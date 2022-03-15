#' @export
TorchOpParallel = R6Class("TorchOpParallel",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "parallel", param_vals = list(), .paths, .reduce = "add") {
      assert(length(.paths) == 2L)
      assert_names(names(.paths), type = "strict")
      assert_choice(.reduce, "add")
      private$.reduce = .reduce
      # 1. unify input --> set everything as a graph
      private$.paths = imap(.paths,
        function(path, name) {
          # assert(inherits(path, "Graph") || inherits(path, "TorchOp"))
          if (test_r6(path, "Graph")) {
            assert_true(all(map_lgl(path$pipeops, function(x) inherits(x, "TorchOp"))))
            graph = path
          } else if (test_r6(path, "TorchOp")) {
            graph = Graph$new()$add_pipeop(path)
          } else {
            stopf("Each path must be a TorchOp or Graph consisting of TorchOps.")
          }
          return(graph)
        }
      )
      param_set = extract_paramset(private$.paths)
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "parallel",
    .paths = NULL,
    .reduce = NULL,
    .build = function(input, param_vals, task) {
      outputs = map(private$.paths,
        function(graph) graph$train(task)
      )
      architectures = map(outputs, function(x) x[[2]])
      outputs = map(architectures, function(x) reduce_architecture(x, task, input))
      # parallel layers = players
      players = map(outputs, "model")
      tensors_out = map(outputs, "output")
      # tensors = map(players, function(layer) with_no_grad(layer$forward(input)))
      # shapes = map(players, function(layer) with_no_grad(layer$forward(input))$shape)
      # assert(length(unique(shapes)) == 1)
      layer = nn_parallel(players[[1L]], players[[2L]], private$.reduce)
      return(layer)
    }
  )
)


nn_parallel = nn_module("parallel",
  initialize = function(path1, path2, reduce) {
    assert_choice(reduce, choices = "add")
    self$path1 = path1
    self$path2 = path2
    self$reduce = switch(reduce,
      add = torch_add,
      stop("Not supported")
    )
  },
  forward = function(input) {
    self$reduce(self$path1(input), self$path2(input))
  }
)

mlr_torchops$add("parallel", TorchOpParallel)
