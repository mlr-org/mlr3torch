#' @title Class for Torch Module Wrappers
#'
#' @name mlr_pipeops_module
#'
#' @description
#' `PipeOpModule` wraps an [`nn_module`] that is being called during the `train` phase of this [`mlr3pipelines::PipeOp`].
#' By doing so, this allows to assemble `PipeOpModule`s in a computational [`mlr3pipelines::Graph`] that
#' represents a neural network architecture. Such a graph can also be used to create a [`nn_graph`] which inherits
#' from [`nn_module`].
#'
#' In most cases it is easier to create such a network by creating a isomorphic graph consisting
#' of nodes of class [`PipeOpTorchIngress`] and [`PipeOpTorch`]. This graph will then generate the graph consisting
#' of `PipeOpModule`s as part of the [`ModelDescriptor`].
#'
#' @section Input and Output Channels:
#' The number and names of the input and output channels can be set during construction. They input and output
#' `"*"` during training, and `NULL` during prediction as the prediction phase currently serves no
#' meaningful purpose.
#'
#' @template pipeop_torch_state_default
#' @section Parameters:
#' No parameters.
#'
#' @section Internals:
#' During training, the wrapped [`nn_module`] is called with the provided inputs in the order in which the channels
#' are defined. Arguments are **not** matched by name.
#'
#' @family Graph Network
#' @family PipeOp
#' @export
#' @examples
#' ## creating an PipeOpModule manually
#'
#' # one input and output channel
#' po_module = PipeOpModule$new("linear",
#'   torch::nn_linear(10, 20),
#'   inname = "input",
#'   outname = "output"
#' )
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
#' po_module = PipeOpModule$new(
#'   "custom",
#'   module,
#'   inname = c("x", "z"),
#'   outname = c("out1", "out2")
#' )
#' x = torch::torch_randn(1, 3)
#' z = torch::torch_randn(1, 3)
#' out = po_module$train(list(x = x, z = z))
#' str(out)
#'
#' # How a PipeOpModule is usually generated
#' graph = po("torch_ingress_num") %>>% po("nn_linear", out_features = 10L)
#' result = graph$train(tsk("iris"))
#' # The PipeOpTorchLinear generates a PipeOpModule and adds it to a new (module) graph
#' result[[1]]$graph
#' linear_module = result[[1L]]$graph$pipeops$nn_linear
#' linear_module
#' formalArgs(linear_module$module)
#' linear_module$input$name
PipeOpModule = R6Class("PipeOpModule",
  inherit = PipeOp,
  public = list(
    #' @field module ([`nn_module`])\cr
    #'   The torch module that is called during the training phase.
    module = NULL,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template param_id
    #' @param module ([`nn_module`])\cr
    #'   The torch module that is being wrapped.
    #' @param inname (`character()`)\cr
    #'   The names of the input channels.
    #' @param outname (`character()`)\cr
    #'   The names of the output channels. If this parameter has length 1, the parameter [module][torch::nn_module] must
    #'   return a [tensor][torch::torch_tensor]. Otherwise it must return a `list()` of tensors of corresponding length.
    #' @template param_param_vals
    #' @template param_packages
    initialize = function(id = "module", module = nn_identity(), inname = "input", outname = "output",
      param_vals = list(), packages = character(0)) {
      private$.multi_output = length(outname) > 1L
      assert(check_class(module, "nn_module"), check_class(module, "function"))
      self$module = module
      assert_names(outname, type = "strict")
      assert_character(packages, any.missing = FALSE)

      input = data.table(name = inname, train = "*", predict = "NULL")
      output = data.table(name = outname, train = "*", predict = "NULL")

      super$initialize(
        id = id,
        input = input,
        output = output,
        param_vals = param_vals,
        packages = packages
      )
    }
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
    .multi_output = FALSE,
    .additional_phash_input = function() {
      # otherwise different nn_linear(1, 1) would get the same hash
      list(address(self$module), self$input$name, self$output$name, self$packages)
    }
  )
)

#' @include zzz.R
register_po("module", PipeOpModule)
