#' @title Class for Torch Module Wrappers
#'
#' @usage NULL
#' @name mlr_pipeops_module
#' @format [`R6Class`] object inheriting from [PipeOp`].
#'
#' @description
#' `PipeOpModule` wraps an [`nn_module`] that is being called during the `train` phase of this [`PipeOp`].
#' By doing so, this allows to assemble `PipeOpModule`s in a computational [graph][`Graph`] that represents a neural
#' network architecture. Such a graph can also be used to create a [`nn_graph`] which inherits from [`nn_module`].
#'
#' In most cases it is easier to create such a network by creating a isomorphic graph consisting
#' of nodes of class [`PipeOpTorchIngress`] and [`PipeOpTorch`]. This graph will then generate the graph consisting
#' of `PipeOpModule`s during its training phase.
#'
#' The `predict` method does currently not serve a meaningful purpose.
#'
#' @section Construction:
#' `r roxy_construction(PipeOpModule)`
#'
#' `r roxy_param_id("module")`
#'
#' * `module` :: [`nn_module`]\cr
#'   The torch module that is being wrapped.
#' * `inname` :: `character()`\cr
#'   The names of the input channels.
#' * `outname` :: `character()`\cr
#'   The names of the output channels. If this parameter has length 1, the parameter [module][torch::nn_module] must
#'   return a [tensor][torch::torch_tensor]. Otherwise it must return a `list()` of tensors of corresponding length.
#'
#' `r roxy_param_param_vals()`
#'
#' `r roxy_param_packages()`
#'
#' @section Input and Output Channels:
#' The number and names of the input and output channels can be set during construction. They input and output
#' `"torch_tensor"` during training, and `NULL` during prediction as the prediction phase currently serves no
#' meaningful purpose.
#'
#' @section State:
#' The `$state` is an empty `list()`.
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Internals:
#' During training, the wrapped [`nn_module`] is called with the provided inputs in the order in which the channels
#' are defined. Arguments are **not** matched by name.
#'
#' @section Fields:
#' * `module` :: `nn_module`\cr
#'   The torch module that is called during the training phase of the PipeOpModule.
#'
#' @section Methods:
#' Only methods inherited from [`PipeOp`].
#'
#' @seealso nn_module, mlr_pipeops_torch, nn_graph, model_descriptor_to_module, PipeOp, Graph
#' @export
#' @examples
#' ## creating an PipeOpModule manually
#'
#' # one input and output channel
#' po_module = PipeOpModule$new("linear", torch::nn_linear(10, 20), inname = "input", outname = "output")
#' x = torch::torch_randn(16, 10)
#' # This calls the forward function of the wrapped module.
#' y = po_module$train(list(input = x))
#' str(y)
#'
#' # multiple input and output channels
#' nn_custom = torch::nn_module("nn_custom",
#'   initialize = function(in_features, out_features) {
#'     self$lin1 = torch::nn_linear(in_features, out_features)
#'     self$lin2 = torch::nn_linear(in_features, out_features)
#'   },
#'   forward = function(x, z) {
#'     list(out1 = self$lin1(x), out2 = torch::nnf_relu(self$lin2(z)))
#'   }
#' )
#'
#' module = nn_custom(3, 2)
#' po_module = PipeOpModule$new("custom", module, inname = c("x", "z"), outname = c("out1", "out2"))
#' x = torch::torch_randn(1, 3)
#' z = torch::torch_randn(1, 3)
#' out = po_module$train(list(x = x, z = z))
#' str(out)
#'
#' # How a PipeOpModule is usually generated
#' graph = pot("ingress_num") %>>% pot("linear", out_features = 10L)
#' result = graph$train(tsk("iris"))
#' # The PipeOpTorchLinear generates a PipeOpModule and adds it to a new graph that represents the architecture
#' result[[1]]$graph
#' linear_module = result[[1L]]$graph$pipeops$linear
#' linear_module
#' formalArgs(linear_module$module)
#' linear_module$input$name
PipeOpModule = R6Class("PipeOpModule",
  inherit = PipeOp,
  public = list(
    module = NULL,
    initialize = function(id, module, inname, outname, param_vals = list(), packages = character(0)) {
      private$.multi_output = length(outname) > 1L
      self$module = assert_class(module, "nn_module")
      lockBinding("module", self)
      assert_names(outname, type = "strict")
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
    }
    # TODO: Maybe implement printer
    # print = function(...) {
    #   output = c(sprintf("<PipeOpModule:%s>", self$id), capture.output(print(self$module, ...)))
    #   output[2] = sub("^An", "Wrapping an",  output[2])
    #   walk(output, function(l) catn(l))
    # }
  ),
  private = list(
    .train = function(inputs) {
      self$state = list()  # PipeOp API requires this.
      # the inputs are passed in the order in which they appear in `graph$input`
      # Note that PipeOpTorch ensures that (unless the forward method has a ... argument) the input channels are
      # identical to the arguments of the forward method.
      outputs = do.call(self$module, unname(inputs))
      outname = self$output$name
      if (!private$.multi_output) outputs = list(outputs)
      outputs
    },
    .predict = function(inputs) {
      rep(list(NULL), nrow(self$output))
    },
    .multi_output = FALSE
  )
)

#' @include zzz.R
register_po("module", PipeOpModule)
