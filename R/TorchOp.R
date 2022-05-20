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

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input,
        output = output,
        packages = packages
      )
    },
    #' @description Builds a Torch Operator
    #'
    #' @param inputs (named `list()`)\cr
    #'   Named list of `torch_tensor`s that form a batch that is the input
    #'   for the current layer. The names have to correspond to the names of the
    #'   [TorchOp's][TorchOp] input channels.
    #' @param task (`mlr3::Task`)\cr
    #'   The task for which to build the architecture.
    #' @param y (`torch_tensor`)\cr
    #'   A batch of the target variable.
    #' @return `torch::nn_module()` where the arguments of the `$forward()` function correspond
    #' to the names of the input channels and the output is a single `torch_tensor`.
    build = function(inputs, task, y) {
      # TODO: Do checks
      if ((length(inputs) > 1L) && is.list(inputs)) { # Merging branches --> all tasks need to be
        # identical
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
        stopf(
          "Forward pass for the layer from '%s' failed for the given input.",
          class(self)[[1L]]
        )
      }

      return(list(layer = layer, output = output))
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
      architecture = inputs[[1L]][["architecture"]]

      architecture$add_torchop(self)

      output = map(
        self$output$name,
        function(channel) {
          structure(
            class = "ModelArgs",
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
    }
  )
)
