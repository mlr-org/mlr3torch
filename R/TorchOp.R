#' @title Abstract Base Class for Torch Operators
#' @description All TorchOps inherit from this class.
#' @export
TorchOp = R6Class("TorchOp",
  inherit = PipeOp,
  public = list(
    #' @description Initializes a new instance of this [R6 Class][R6::R6Class].
    #' @template param_id
    #' @param param_set (`ParamSet`)\cr
    #'   Parameter set to be set for the [PipeOp][mlr3pipelines::PipeOp].
    #' @param param_vals (named `list()`)\cr
    #'   Named list with parameter values to be set after construction.
    #' @param input (`data.table()`)\cr
    #'   Input channels to be set for the [PipeOp][mlr3pipelines::PipeOp].
    #'   The input default name is "input", accpets "ModelArgs" during `$train()` and a "Task"
    #'   during `$predict()`.
    #' @param output (`data.table()`)\cr
    #'   Output channels to be set for the [PipeOp][mlr3pipelines::PipeOp].
    #'   The output default name is "output", accepts "ModelArgs" during `$train()` and a "Task"
    #'   during `$predict()`.
    #' @param packages (`character()`) The packages on which the [TorchOp][TorchOp] depends.
    initialize = function(id, param_set, param_vals, input = NULL, output = NULL, packages = NULL) {
      # default input and output channels, packages
      input = input %??% data.table(name = "input", train = "ModelArgs", predict = "Task")
      output = output %??% data.table(name = "output", train = "ModelArgs", predict = "Task")
      packages = packages %??% c("torch", "mlr3torch")
      assert_true(id != "repeat") # this leads to bugs

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input,
        output = output,
        packages = packages
      )
    },
    #' @description
    #' Builds a Torch Operator. This is not applicable to all TorchOp's, exceptions are:
    #'  * TorchOpOptimizer
    #'  * TorchOpLoss
    #'  * TorchOpModel
    #'  * TorchOpBlock
    #'
    #' @param inputs (named `list()`)\cr
    #'   Named list of `torch_tensor`s that form a batch that is the input
    #'   for the current layer. The names have to correspond to the names of the
    #'   [TorchOp's][TorchOp] input channels.
    #' @param task (`mlr3::Task`)\cr
    #'   The task for which to build the network.
    #' @return `torch::nn_module()` where the arguments of the `$forward()` function correspond
    #' to the names of the input channels and the output is a single `torch_tensor`.
    build = function(inputs, task) {
      if ((length(inputs) > 1L) && is.list(inputs)) { # Merging branches --> all tasks need to be
        # identical
        hashes = map(map(inputs, "task"), "hash")
        assert_true(length(unique(hashes)) == 1L)
      }

      layer = private$.build(inputs, task)
      # not exactly sue why we have to do this, but otherwise there were bugs with the
      # running_var in the batch_norm1d being -nan
      layer$eval()
      output = try(with_no_grad(invoke(layer$forward, .args = inputs)), silent = TRUE)
      layer$train()
      # otherwise batch_norm had some weird bug
      reset_running_stats(layer)

      if (inherits(output, "try-error")) {
        stopf(
          "Forward pass for the layer from '%s' failed for the given input.\nMessage: %s",
          class(self)[[1L]], output
        )
      }

      if (inherits(output, "torch_tensor")) {
        output = set_names(list(output), self$output$name)
      } else {
        assert_list(output)
        assert_set_equal(self$output$name, names(output))
      }

      list(layer = layer, output = output)
    },
    #' @description Printer for this object.
    #' @param ... (any)\cr
    #'   Currently unused.
    print = function(...) {
      type_table_printout = function(table) {
        strings = do.call(sprintf, cbind(fmt = "%s`[%s,%s]", table[, c("name", "train", "predict")]))
        strings = strwrap(paste(strings, collapse = ", "), indent = 2, exdent = 2)
        if (length(strings) > 6) {
          strings = c(strings[1:5], sprintf("  [... (%s lines omitted)]", length(strings) - 5))
        }
        gsub("`", " ", paste(strings, collapse = "\n"))
      }

      catf("TorchOp: <%s> (%strained)", self$id, if (self$is_trained) "" else "not ")
      catf("values: <%s>", as_short_string(self$param_set$values))
      catf("Input channels <name [train type, predict type]>:\n%s", type_table_printout(self$input))
      catf("Output channels <name [train type, predict type]>:\n%s", type_table_printout(self$output))
    }
  ),
  private = list(
    .train = function(inputs) {
      task = inputs[[1L]][["task"]]
      network = inputs[[1L]][["network"]]

      pv = self$param_set$get_values(tags = "train")
      layer_inputs = map(inputs, "output")
      c(layer, outputs) %<-% self$build(layer_inputs, task)

      network$add_layer(self$id, layer, self$input$name, self$output$name)
      iwalk(
        inputs,
        function(input, nm) {
          network$add_edge(
            src_id = input$id,
            src_channel = input[["channel"]],
            dst_id = self$id,
            dst_channel = nm
          )
        }
      )

      outputs = map(
        self$output$name,
        function(channel) {
          structure(
            class = "ModelArgs",
            list(task = task, network = network, channel = channel, id = self$id,
              output = outputs[[channel]]
            )
          )
        }
      )
      self$state = list()
      set_names(outputs, self$output$name)

      return(outputs)
    },
    .predict = function(inputs) {
      inputs
    },
    .build = function(inputs, task) {
      stopf("Private method $.build() not applicable to this TorchOp.")
    }
  )
)
