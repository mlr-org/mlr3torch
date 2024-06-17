#' @title Block Repetition
#' @description
#' Repeat a block n times.
#' @section Parameters:
#' The parameters available for the block itself, as well as
#' * `times` :: `integer(1)`\cr
#'   How often to repeat the block.
#' @section Input and Output Channels:
#' The `PipeOp` sets its input and output channels to those from the `block` (Graph)
#' it received during construction.
#' @templateVar id nn_block
#' @template pipeop_torch
#' @export
#' @examples
#' block = po("nn_linear") %>>% po("nn_relu")
#' po_block = po("nn_block", block,
#' nn_linear.out_features = 10L, times = 3)
#' network = po("torch_ingress_num") %>>%
#' po_block %>>%
#' po("nn_head") %>>%
#' po("torch_loss", t_loss("cross_entropy")) %>>%
#' po("torch_optimizer", t_opt("adam")) %>>%
#' po("torch_model_classif",
#'   batch_size = 50,
#'   epochs = 3)
#'
#' task = tsk("iris")
#' network$train(task)
PipeOpTorchBlock = R6Class("PipeOpTorchBlock",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template param_id
    #' @template param_param_vals
    #' @param block ([`Graph`])\cr
    #'   A graph consisting primarily of [`PipeOpTorch`] objects that is to be
    #'   repeated.
    initialize = function(block, id = "nn_block", param_vals = list()) {
      private$.block = as_graph(block)
      private$.param_set_base = ps(
        times = p_int(lower = 1L, tags = c("train", "required"))
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
    #' @field block ([`Graph`])\cr
    #' The neural network segment that is repeated by this `PipeOp`.
    block = function(rhs) {
      assert_ro_binding(rhs)
      private$.block
    }
  ),
  private = list(
    .block = NULL,
    .make_graph = function(block, times) {
      graph = block
      for (i in seq_len(times - 1L)) {
        block = clone_graph_unique_ids(block)
        graph = graph %>>% block
      }
      graph
    },
    .shapes_out = function(shapes_in, param_vals, task)  {
      if (is.null(task)) {
        stopf("PipeOpTorchBlock '%s', requires a task to compute output shapes", self$id)
      }
      block = private$.block$clone(deep = TRUE)
      walk(block$pipeops, function(po) {
        if (test_class(po, "PipeOpTorch")) {
          get_private(po, ".only_shape") = TRUE
        }
      })
      # thereby we avoid initializing the nn modules (it is a little hacky)
      graph = private$.make_graph(block, param_vals$times)

      mds = map(seq_along(shapes_in), function(i) {
        ModelDescriptor(
          # because we set the .only_shape above, the graph is not touched
          graph = as_graph(po("nop", id = paste0("nop.", i))),
          ingress = set_names(list(
            TorchIngressToken(
              features = "placeholder",
              batchgetter = function(data, device, ...) NULL,
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
      param_vals = self$param_set$get_values(tags = "train")
      block = private$.block$clone(deep = TRUE)
      graph = private$.make_graph(block, param_vals$times)
      out = graph$train(inputs, single_input = FALSE)
    },
    .param_set_base = NULL
  )
)


#' @include zzz.R
register_po("nn_block", PipeOpTorchBlock, metainf = list(block = as_graph(po("nop"))))
