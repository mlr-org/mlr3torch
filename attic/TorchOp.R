#' @title Abstract Base Class for Torch Operators
#'
#' @description
#' `TorchOp` can be combined in a [mlr3pipelines::Graph] to obtain a - custom, yet fully
#' parameterized - neural network learner.
#'
#' @details
#'
#' Internally, this configuration is stored in a list with the class `"ModelConfig"` that
#' can be seen as the input to [LearnerTorchClassif] / [LearnerTorchRegr].
#' The parameter set for [TorchOpModel] in combination with the ModelConfig define all
#' parameters to train a [LearnerClassifTorch] or [LearnerRegrTorch].
#'
#' The ModelConfig is initialized by [TorchOpInput].
#' The ModelConfig is executed using [TorchOpModel].
#' All other [TorchOp]s modify the ModelConfig.
#'
#'
#' All other TorchOps modify the ModelConfig, see section *Building the Network*
#'
#' Those modifications are threefold:
#'
#'   * Modify the network represented as a [nn_graph]
#'   * Configure the optimizer through TorchOpOptimizer
#'   * Configure the loss through TorchOpLoss
#'
#' @section Building the Network:
#'
#'
#' @section Network modifiers:
#' Each TorchOp that has a private `.build` function implemented modifies the architecture of the
#' neural network. To do so, it has to do two things:
#' 1. Build a `"nn_module"` using it's input, the task for which it is being built and the parameters
#' that are set.
#' 2. Connect the (id_prev, out_channel_prev) that create its input to its respective
#' (id_current, in_channel_current)
#'
#' The edges of this graph network are build by a message-passing mechanism.
#' I.e. when a TorchOp outputs a ModelConfig to it's various channel, it always includes its own
#' id and the output of the channel. This allows the next TorchOp to not add the edges to the
#' graph network.
#'
#' For more details see [nn_graph].
#'
#' @section ModelConfig:
#' An object of class `"ModelConfig"` is a `list()` with the class attribute set to `"ModelConfig"`.
#' It is only used internally, hence there is no constructor provided.
#' A ModelConfig
#'
#' @section Special TorchOps:
#' TorchOpOptimizer, TorchOpLoss, TorchOpInput and TorchOpOutputy, TorchOpRepeat.
#'
#' @examples
#' task = tsk("iris")
#' top("input")$train(list(task))$output
#'
#' graph = top("input") %>>%
#'   top("select", items = "num") %>>%
#'   top("linear", out_features = 10L) %>>%
#'   top("relu") %>>%
#'   top("output") %>>%
#'   top("optimizer", optimizer = "adam", lr = 0.1) %>>%
#'   top("loss", "cross_entropy") %>>%
#'   top("model.classif", batch_size = 16L, epochs = 1L)
#'
#' learner = as_learner(graph)
#' # we can modify the architecture
#' learner
#' learner$param_set$values$linear.out_features = 20L
#'
#' learner$train(task)
#'
#'
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
    #'   The input default name is "input", accpets "ModelConfig" during `$train()` and a "Task"
    #'   during `$predict()`.
    #' @param output (`data.table()`)\cr
    #'   Output channels to be set for the [PipeOp][mlr3pipelines::PipeOp].
    #'   The output default name is "output", accepts "ModelConfig" during `$train()` and a "Task"
    #'   during `$predict()`.
    #' @param packages (`character()`) The packages on which the [TorchOp][TorchOp] depends.
    initialize = function(id, param_set, param_vals, input = NULL, output = NULL, packages = NULL) {
      # default input and output channels, packages
      input = input %??% data.table(name = "input", train = "ModelConfig", predict = "Task")
      output = output %??% data.table(name = "output", train = "ModelConfig", predict = "Task")
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
        if (is.null(names(output))) {
          names(output) = self$output$name
        } else {
          assert_set_equal(self$output$name, names(output))
        }
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
            class = "ModelConfig",
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
