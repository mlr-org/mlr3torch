# This test file does not ues the autotest for the torchop, because here we only test the functionality that
# remains unchanged for the child classes.

nn_debug = nn_module(
  initialize = function(d_in1, d_in2, d_out1, d_out2, bias) {
    self$linear1 = nn_linear(d_in1, d_out1, bias)
    self$linear2 = nn_linear(d_in2, d_out2, bias)
  },
  forward = function(input1, input2) {
    output1 = self$linear1(input1)
    output2 = self$linear1(input2)

    list(output1 = output1, output2 = putput2)
  }
)

PipeOpTorchDebug = R6Class("PipeOpTorchDebug",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_debug", param_vals = list(), inname, outname) {
      param_set = ps(
        d_out1 = p_int(lower = 1, tags = "required"),
        d_out2 = p_int(lower = 1, tags = "required"),
        bias = p_lgl()
      )
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        inname = inname,
        outname = outname,
        module_generator = nn_debug
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals) {
      c(param_vals, list(d_in1 = tail(shapes_in[["input1"]], 1)), d_in2 = tail(shapes_in[["input2"]], 1))
    },
    .shapes_out = function(shapes_in, param_vals) {
      list(
        input1 = c(head(shapes_in[["input1"]][[1]], -1), param_vals$d_out1),
        input2 = c(head(shapes_in[["input2"]][[1]], -1), param_vals$d_out2)
      )
    }
  )
)

test_that("PipeOpTorch works", {
  task = tsk("penguins")

  # basic checks that output is checked correctly
  expect_error(
    PipeOpTorchDebug$new(id = "debug", inname = paste0("input", 1:2), outname = paste0("output", 1:2)),
    regex = "grepl"
  )
  expect_error(
    PipeOpTorchDebug$new(id = "nn_debug", inname = "input", outname = paste0("output", 1:2)),
    regex = "The names of the input channels must be a permutation of the arguments of the provided module_generator."
  )
  expect_error(
    PipeOpTorchDebug$new(id = "nn_debug", inname = "input", outname = paste0("output", 1:2)),
    regex = "The names of the input channels must be a permutation of the arguments of the provided module_generator."
  )

  expect_error(
    PipeOpTorchDebug$new(id = "nn_debug", inname = "input", outname = paste0("output", 1:2)),
    regex = "The names of the input channels must be a permutation of the arguments of the provided module_generator."
  )


  # basic checks that the channels are set correctly
  obj = PipeOpTorchDebug$new(id = "nn_debug", inname = paste0("input", 1:2), outname = paste0("output", 1:2))
  obj$param_set$values = list(d_out1 = 2, d_out2 = 3, bias = TRUE)

  mdin1 = pot("ingress_num")$train(list(task))[[1L]]
  mdin2 = pot("ingress_cat")$train(list(task))[[1L]]

  mdouts = obj$train(list(input1 = mdin1, input2 = mdin2))
  mdout1 = mdouts[["output1"]]

  expect_true(identical(mdout1$graph, mdout2$graph))
  expect_true(identical(mdout1$task, mdout2$task))

  expect_true(mdout1$id == "")
  mdout2 = mdouts[["output2"]]


  expect_true(list)

  exp

  expect_error(graph)

})
