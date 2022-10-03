#' @title Class for Torch Module Wrappers
#'
#' @usage NULL
#' @name mlr_pipeops_module
#' @format [`R6Class`] object inheriting from [PipeOp`].
#'
#' @description
#' `PipeOpModule` wraps a `torch::nn_module`, which is also called during training.
#' The prediciton p
#' In the training phase, it calls the w
#'
#' Graphs of `PipeOpModule`s can be used as a graph in [`nn_graph`].
#' Usually, a [`PipeOpModule`] is generated during the training phase of a [`PipeOpTorch`] as shown in the examples.
#' During training, this pipeop simply calls the underlying [`module`][torch::nn_module].
#' During prediction, it does nothing.
#'
#' @details
#' Currently, there is not a perfect correspondence between what a [PipeOp] is inteded for and what PipeOpModule does.
#'
#' @section Input and Output Channels:
#' The input channels correspond to the arguments of the wrapped [module][torch::nn_module].
#' If there is more than one output, the output names are
#'
#'
#' The output is the input [`Task`][mlr3::Task] with all affected numeric features replaced by their
#' non-negative components.
#'
#' #'
#' @section State:
#' The `$state` is an empty list().
#'
#' @section Parameters:
#' This [PipeOp] does not have any parameters.
#'
#'
#' @template param_id
#' @template param_param_vals
#' @template param_packages
#' @param inname (`character()`)\cr
#'   The names of the input channels, must correspond to the argument names of the wrapped [torch::nn_module].
#' @param outname (`character()`)
#'   The names of the output channels. If the [module][torch::nn_module] returns a list of tensors, this must be a
#'   permutation of the names of the returned list.
#'   If there is only one output channel, the underlying [module][torch::nn_module] must return a
#'   [tensor][torch::torch_tensor] and the outname can be selected.
#'
#' @examples
if (FALSE) {
  graph = pot("ingress_num") %>>% pot("linear", out_features = 10L)
  result = graph$train(tsk("iris"))
  linear_module = result[[1L]]$graph$pipeops$linear
  linear_module
}
#'
#' formalArgs(linear_module$module)
#' linear$module$input$name
#'
#' @export
PipeOpModule = R6Class("PipeOpModule",
  inherit = PipeOp,
  public = list(
    module = NULL,
    #' @description Initializes a new instance of this [R6 Class][R6::R6Class].
    #' @param multi_input (`NULL` | `integer(1)`)\cr
    #'   `0`: `...`-input. Otherwise: `multi_input` times input channel named `input1:`...`input#`.\cr
    #'   `module`'s `$forward` function must take `...`-input if `multi_input` is 0, and must have `multi_input`
    #'   arguments otherwise.
    #' @param multi_output (`NULL` | `integer(1)`)\cr
    #'   `NULL`: single output. Otherwise: `multi_output` times output channel named `output1:`...`input#`.\cr
    #'   `module`'s `$forward` function must return a `list` of `torch_tensor` if `multi_output` is not `NULL`.
    initialize = function(id, module, inname, outname, param_vals = list(),
      packages = character(0)) {
      self$module = assert_class(module, "nn_module")
      lockBinding("module", self)
      assert_names(inname, type = "strict")
      assert_names(outname, type = "strict")
      assert_true(all(sort(inname) == formalArgs(module)))
      assert_character(packages, any.missing = FALSE)

      input = data.table(name = inname, train = "torch_tensor", predict = "NULL")
      output = data.table(name = outname, train = "torch_tensor", predict = "NULL")

      super$initialize(
        id = id,
        input = input,
        output = output,
        param_vals = param_vals,
        packages = packages
      )
    },
    print = function(...) {
      output = c(sprintf("<PipeOpModule:%s>", self$id), capture.output(print(self$module)))
      output[2] = sub("^An", "Wrapping an",  output[2])
      walk(output, function(l) catn(l))
    }
  ),
  private = list(
    .train = function(inputs) {
      self$state = list()  # PipeOp API requires this.
      outputs = do.call(self$module, inputs)
      outname = self$output$name
      if (private$.multi_output) outputs = list(outname = outputs)  # the only case where module does not produce a list
      outputs
    },
    .predict = function(inputs) {
      rep(list(NULL), nrow(self$output))
    },
    .multi_output = FALSE
  )
)

