#' @title Class for Torch Module Wrappers
#'
#' @name mlr_pipeops_module
#'
#' @description
#' `PipeOpModule` wraps an [`nn_module`][torch::nn_module] or `function` that is being called during the `train` phase of this
#' [`mlr3pipelines::PipeOp`]. By doing so, this allows to assemble `PipeOpModule`s in a computational
#' [`mlr3pipelines::Graph`] that represents either a neural network or a preprocessing graph of a [`lazy_tensor`].
#' In most cases it is easier to create such a network by creating a graph that generates this graph.
#'
#' In most cases it is easier to create such a network by creating a structurally related graph consisting
#' of nodes of class [`PipeOpTorchIngress`] and [`PipeOpTorch`]. This graph will then generate the graph consisting
#' of `PipeOpModule`s as part of the [`ModelDescriptor`].
#'
#' @section Input and Output Channels:
#' The number and names of the input and output channels can be set during construction. They input and output
#' `"torch_tensor"` during training, and `NULL` during prediction as the prediction phase currently serves no
#' meaningful purpose.
#'
#' @template pipeop_torch_state_default
#' @section Parameters:
#' No parameters.
#'
#' @section Internals:
#' During training, the wrapped [`nn_module`][torch::nn_module] / `function` is called with the provided inputs in the order in which
#' the channels are defined. Arguments are **not** matched by name.
#'
#' @family Graph Network
#' @family PipeOp
#' @export
#' @examplesIf torch::torch_is_installed()
#' ## creating an PipeOpModule manually
#'
#' # one input and output channel
#' po_module = po("module",
#'   id = "linear",
#'   module = torch::nn_linear(10, 20),
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
#' po_module = po("module",
#'   id = "custom",
#'   module = module,
#'   inname = c("x", "z"),
#'   outname = c("out1", "out2")
#' )
#' x = torch::torch_randn(1, 3)
#' z = torch::torch_randn(1, 3)
#' out = po_module$train(list(x = x, z = z))
#' str(out)
#'
#' # How such a PipeOpModule is usually generated
#' graph = po("torch_ingress_num") %>>% po("nn_linear", out_features = 10L)
#' result = graph$train(tsk("iris"))
#' # The PipeOpTorchLinear generates a PipeOpModule and adds it to a new (module) graph
#' result[[1]]$graph
#' linear_module = result[[1L]]$graph$pipeops$nn_linear
#' linear_module
#' formalArgs(linear_module$module)
#' linear_module$input$name
#'
#' # Constructing a PipeOpModule using a simple function
#' po_add1 = po("module",
#'   id = "add_one",
#'   module = function(x) x + 1
#' )
#' input = list(torch_tensor(1))
#' po_add1$train(input)$output
PipeOpModule = R6Class("PipeOpModule",
  inherit = PipeOp,
  public = list(
    #' @field module ([`nn_module`][torch::nn_module])\cr
    #'   The torch module that is called during the training phase.
    module = NULL,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template param_id
    #' @param module ([`nn_module`][torch::nn_module] or `function()`)\cr
    #'   The torch module or function that is being wrapped.
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
      assert(check_class(module, "nn_module"), check_class(module, "function"), combine = "or")
      self$module = module
      packages = union(c("mlr3torch", "torch"), packages)

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
  ),
  private = list(
    .train = function(inputs) {
      self$state = list()  # PipeOp API requires this.
      outputs = do.call(self$module, unname(inputs))
      outname = self$output$name
      if (!private$.multi_output) outputs = list(outputs)
      outputs
    },
    .predict = function(inputs) {
      rep(list(NULL), nrow(self$output))
    },
    .multi_output = NULL,
    .additional_phash_input = function() {
      # mlr3pipelines does not use calculate_hash, but calls directly into digest, hence we have to take
      # care of the byte code
      fn_input = if (test_class(self$module, "nn_module")) {
        address(self$module)
      } else {
        list(formals(self$module), body(self$module), address(environment(self$module)))
      }

      list(fn_input, self$input$name, self$output$name, self$packages)
    },
    deep_clone = function(name, value) {
      if (name == "module" && test_class(value, "nn_module")) {
        value$clone(deep = TRUE)
      } else {
        super$deep_clone(name, value)
      }
    }
  )
)

#' @include zzz.R
register_po("module", PipeOpModule)
