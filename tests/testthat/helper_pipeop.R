nn_debug = nn_module(
  initialize = function(d_in1, d_in2, d_out1, d_out2, bias = TRUE) {
    self$linear1 = nn_linear(d_in1, d_out1, bias)
    self$linear2 = nn_linear(d_in2, d_out2, bias)
  },
  forward = function(input1, input2) {
    output1 = self$linear1(input1)
    output2 = self$linear2(input2)

    list(output1 = output1, output2 = output2)
  }
)

PipeOpTorchDebug = R6Class("PipeOpTorchDebug",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_debug", param_vals = list(), inname = c("input1", "input2"),
      outname = c("output1", "output2")) {
      param_set = ps(
        d_out1 = p_int(lower = 1, tags = c("required", "train")),
        d_out2 = p_int(lower = 1, tags = c("required", "train")),
        bias = p_lgl(default = TRUE, tags = "train")
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
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals, list(d_in1 = tail(shapes_in[["input1"]], 1)), d_in2 = tail(shapes_in[["input2"]], 1))
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      list(
        input1 = c(head(shapes_in[[1]], -1), param_vals$d_out1),
        input2 = c(head(shapes_in[[2]], -1), param_vals$d_out2)
      )
    }
  )
)

PipeOpPreprocTorchAddSome = pipeop_preproc_torch("trafo_some",
  param_set = ps(some = p_dbl(default = 1L, tags = "train")),
  fn = crate(function(x, some = 1L) x + some),
  shapes_out = "infer"
)
