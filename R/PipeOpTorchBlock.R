#' @title Block Repetition
#' @description
#' Repeat a block `n_blocks` times by concatenating it with itself (via `%>>%`).
#' @section Naming:
#' For the generated module graph, the IDs of the modules are generated by prefixing the
#' IDs of the `n_blocks` layers with the ID of the `PipeOpTorchBlock` and postfixing them with
#' `__<layer>`.
#'
#' @section Parameters:
#' The parameters available for the provided `block`, as well as
#' * `n_blocks` :: `integer(1)`\cr
#'   How often to repeat the block.
#' * `trafo` :: `function(i, param_vals, param_set) -> list()`\cr
#'   A function that allows to transform the parameters vaues of each layer (`block`).
#'   Here,
#'   * `i` :: `integer(1)`\cr
#'       is the index of the layer, ranging from `1` to `n_blocks`.
#'   * `param_vals` :: named `list()`\cr
#'       are the parameter values of the layer `i`.
#'   * `param_set` :: [`ParamSet`][paradox::ParamSet]\cr
#'       is the parameter set of the whole `PipeOpTorchBlock`.
#'
#'   The function must return the modified parameter values for the given layer.
#'   This, e.g., allows for special behavior of the first or last layer.
#' @section Input and Output Channels:
#' The `PipeOp` sets its input and output channels to those from the `block` (Graph)
#' it received during construction.
#' @templateVar id nn_block
#' @template pipeop_torch
#' @export
#' @examplesIf torch::torch_is_installed()
#' # repeat a simple linear layer with ReLU activation 3 times, but set the bias for the last
#' # layer to `FALSE`
#' block = nn("linear") %>>% nn("relu")
#'
#' blocks = nn("block", block,
#'   linear.out_features = 10L, linear.bias = TRUE, n_blocks = 3,
#'   trafo = function(i, param_vals, param_set) {
#'     if (i  == param_set$get_values()$n_blocks) {
#'       param_vals$linear.bias = FALSE
#'     }
#'     param_vals
#'   })
#' graph = po("torch_ingress_num") %>>%
#'   blocks %>>%
#'   nn("head")
#' md = graph$train(tsk("iris"))[[1L]]
#' network = model_descriptor_to_module(md)
#' network
PipeOpTorchBlock = R6Class("PipeOpTorchBlock",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template param_id
    #' @template param_param_vals
    #' @param block ([`Graph`][mlr3pipelines::Graph])\cr
    #'   A graph consisting primarily of [`PipeOpTorch`] objects that is to be
    #'   repeated.
    initialize = function(block, id = "nn_block", param_vals = list()) {
      private$.block = as_graph(block)
      private$.param_set_base = ps(
        n_blocks = p_int(lower = 0L, tags = c("train", "required")),
        trafo = p_uty(tags = "train", custom_check = crate(function(x) {
          check_function(x, args = c("i", "param_vals", "param_set"))
        }))
      )

      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = alist(private$.param_set_base, private$.block$param_set),
        inname = private$.block$input$name,
        outname = private$.block$output$name,
        packages = private$.block$packages,
        module_generator = NULL
      )
    }
  ),
  active = list(
    #' @field block ([`Graph`][mlr3pipelines::Graph])\cr
    #' The neural network segment that is repeated by this `PipeOp`.
    block = function(rhs) {
      assert_ro_binding(rhs)
      private$.block
    }
  ),
  private = list(
    .block = NULL,
    .make_graph = function(block, n_blocks) {
      trafo = self$param_set$get_values()$trafo
      graph = block
      graphs = c(replicate(n_blocks, graph$clone(deep = TRUE)))
      if (!is.null(trafo)) {
        param_vals = map(graphs, function(graph) graph$param_set$get_values())
        walk(seq_along(param_vals), function(i) {
          vals = trafo(i = i, param_vals = param_vals[[i]], param_set = self$param_set)
          graphs[[i]]$param_set$values = vals
        })
      }
      lapply(seq_len(n_blocks), function(i) {
        graphs[[i]]$update_ids(prefix = paste0(self$id, "."), postfix = paste0("__", i))
      })
      Reduce(`%>>%`, graphs)
    },
    .shapes_out = function(shapes_in, param_vals, task)  {
      if (is.null(task)) {
        stopf("PipeOpTorchBlock '%s', requires a task to compute output shapes", self$id)
      }
      block = private$.block$clone(deep = TRUE)
      walk(block$pipeops, function(po) {
        # thereby we avoid initializing the nn modules (it is a little hacky)
        if (test_class(po, "PipeOpTorch")) {
          get_private(po, ".only_shape") = TRUE
        }
      })
      graph = private$.make_graph(block, param_vals$n_blocks)

      mds = map(seq_along(shapes_in), function(i) {
        ModelDescriptor(
          # because we set the .only_shape above, the graph is not used at all
          # so we just set it to something
          graph = as_graph(po("nop", id = paste0("nop.", i))),
          ingress = set_names(list(
            TorchIngressToken(
              features = "placeholder",
              batchgetter = function(data, ...) NULL,
              shape = shapes_in[[1]])),
              paste0("nop.", i, ".input")
          ),
          task = task,
          pointer = c(paste0("nop.", i), "output"),
          pointer_shape = shapes_in[[i]]
        )
      })

      mdouts = graph$train(mds, single_input = FALSE)

      map(mdouts, "pointer_shape")
    },
    .train = function(inputs) {
      param_vals = self$param_set$get_values()
      if (param_vals$n_blocks == 0L) {
        return(inputs)
      }
      block = private$.block$clone(deep = TRUE)
      graph = private$.make_graph(block, param_vals$n_blocks)
      inputs = set_names(inputs, graph$input$name)
      graph$train(inputs, single_input = FALSE)
    },
    .param_set_base = NULL,
    .additional_phash_input = function() {
      self$block$phash
    }
  )
)


#' @include aaa.R
register_po("nn_block", PipeOpTorchBlock, metainf = list(block = as_graph(po("nop"))))
