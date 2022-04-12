#' @title Abstract Base Class for Torch Operators
#' @description All TorchOps inherit from this class.
#' @export
TorchOp = R6Class("TorchOp",
  inherit = PipeOp,
  public = list(
    initialize = function(id, param_set, param_vals, input = NULL, output = NULL, packages = NULL) {
      # default input and output channels, packages
      input = input %??% data.table(name = "input", train = "ModelArgs", predict = "Task")
      output = output %??% data.table(name = "output", train = "ModelArgs", predict = "Task")
      packages = packages %??% c("torch", "mlr3torch")

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input,
        output = output,
        packages = packages
      )
    },
    #' @description Builds the torch layer
    #' @param input (torch_tensor) The torch tensor that is the input to this layer.
    #' @param param_vals (list()) parameter values passed to the build function.
    #' @param task (mlr3::Task) The task on which the architecture is trained.
    build = function(inputs, task, y) {
      # TODO: Do checks
      if (length(inputs) > 1L) {
        hashes = map(map(inputs, "task"), "hash")
        assert_true(length(unique(hashes)) == 1L)
      }

      param_vals = self$param_set$get_values(tag = "train")
      layer = private$.build(
        inputs = inputs,
        param_vals = param_vals,
        task = task,
        y = y
      )
      output = try(with_no_grad(do.call(layer$forward, args = inputs)), silent = TRUE)

      if (inherits(output, "try-error")) {
        stopf("Forward pass on the created layer (%s) failed with the given input.",
          private$.operator
        )
      }

      return(list(layer = layer, output = output))
    },
    #' Provides the repreesntation for the TorchOp.
    repr = function() {
      sprinf("<%s: %s>", self$.operator, self$param_set$get_values(tag = "train"))
    }
  ),
  active = list(
    .operator = function() {
      # TODO: This is a bit suspicious
      formals(self$initialize)[["id"]]
    }
  ),
  private = list(
    .train = function(inputs) {
      task = inputs[[1L]][["task"]]
      architecture = inputs[[1L]][["architecture"]]

      architecture$add_torchop(self)

      output = map(
        self$output$name,
        function(channel) {
          structure(class = "ModelArgs",
            list(task = task, architecture = architecture, channel = channel, id = self$id)
          )
        }
      )
      self$state = list()
      set_names(output, self$output$name)

      input_channels = names(inputs)
      for (i in seq_along(inputs)) {
        input = inputs[[i]]
        if (!is.null(input$id)) {
          architecture$add_edge(
            src_id = input$id,
            src_channel = input[["channel"]],
            dst_id = self$id,
            dst_channel = input_channels[[i]]
          )
        }
      }

      return(output)
    },
    .predict = function(inputs) {
      inputs
    },
    .build = function(inputs, param_vals, task, y) {
      stop("ABC")
    },
    .output_dim = function(input_dim) {
      stop("ABC")
    }
  )
)
