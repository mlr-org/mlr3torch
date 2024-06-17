#' @title Block Repetition
#' @description
#' Repeat a block n times.
#' @export
PipeOpTorchBlock = R6Class("PipeOpTorchBlock",
  inherit = PipeOp,
  public = list(
    initialize = function(block, id = "nn_block", param_vals = list()) {
      private$.block = assert_graph(block)
      block2 = block$clone(deep = TRUE)
      if (inherits(try(block2 %>>% block2, silent = TRUE), "try-error")) {
        stopf("PipeOp '%s': Argument block must be chainable with itself.", self$id)
      }
      param_set = psc(ps(
        times = p_int(lower = 1L, tags = c("train", "required"))
      ), private$.block)

      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        input = block$input[, c("name", "train", "predict")],
        output = block$output[, c("name", "train", "predict")],
        packages = "mlr3torch",
        input = data.table()
      )
    }
  ),
  active = list(
    block = function(rhs) {
      assert_ro_binding(rhs)
      private$.block
    }
  ),
  private = list(
    .block = NULL,
    .shapes_out = function(shapes_in, param_vals, task)  {
      block = private$.block$clone(deep = TRUE)
      graph = block
      for (i in seq_len(param_vals$times - 1L)) {
        graph = graph %>>% block
      }
      # create model descriptors

      

      # this needs to create a ModelDescriptor and then call train (without building up the network)
      # and then output the final shape

      # TODO: this might be tricky
    },
    .make_module = function(shapes_in) {
      block = private$block$clone(deep = TRUE)
     
    }

    
  )
)
