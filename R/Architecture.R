#' @title Neural Network (Graph-) Architecture
#' @description Instances of this class represent abstract architectures that can be built from
#' a Task. The nodes are the ids and the edges is the data-flow.
#'
#' The nodes and edges are stored in an environment. This makes it possible to have different
#' shallow copies (with different pointer) in the dataflow induced by training the graph.
#' The nodes and edges are stored in an environment, because due to the TorchOpParallel [name?]
#' operator there can be multiple shallow copies of this architecture that are identical except
#' for the ptr field that is the current output of the architecture. Because of this they
#' are stored in environments that can be simultaneously modified through all the shallow copies.
#'
#' @section Difference to [mlr3pipelines::Graph]
#'  The difference to a Graph is that there is only always one edge between nodes, therefore
#'  one das not have to bother with different channels.
#'
Architecture = R6Class("Architecture",
  inherit = Graph,
  public = list(
    add_node = function(op) {
      self$add_pipeop(op)
    },
    build = function(task, input = NULL) {
      reduction = architecture_reduce(self, task, input)
      edges = simplify_graph(reduction$edges)
      nn_graph_network$new(edges, reduction$layers)
    }
  )
)

#' @title Reduce an Architecture to Layers (nn_modules)
#' @description A Architecture contains builders (nodes) and edges that indicate how the data will
#' flow through the layers that are constructed by the builders.
#' This function builds all the layers.
#' For that the ids of the grpah need to be topologically sorted and then one has to loop over the
#' topologically sorted layers and call the build function. The outputs of each layer are safed
#' until all the direct children of the node are built.
#'
#' This structure is essentially identical to the network_forward function only that we here
#' additionally call the build function and also that the task is passe == idxd instead of the actual
#' network input as obtained from the dataloader.
#'
#' @param architecture (mlr3torch::Architecture) The architecture that is being build from the task.
#' @param task (mlr3::Task) The task for which to build the architecture.
#' @param input (torch::Tensor) The input tensor. If NULL the output of the data-loader that is
#' generated from the task is used. (E.g. used when reducing a block)

#' To reduce an architecture we need the input for the orphan

architecture_reduce = function(self, task = NULL, input = NULL) {
  graph_input = self$input
  graph_output = self$output
  assert(
    test_true(nrow(graph_input) == 1L),
    test_true(nrow(graph_output) == 2L)
  )
  layers = list()

  instance = get_instance(task)
  y = instance[["y"]]
  if (is.null(input)) {
    instance[["y"]] = NULL
    input = instance
  } else {
    if (!inherits(inut, "torch_tensor")) {
      input = list(x = input)
    }
  }

  edges = copy(self$edges)
  edges = rbind(edges,
    data.table(src_id = "__initial__", src_channel = graph_input$name,
      dst_id = graph_input$op.id, dst_channel = graph_input$channel.name),
    data.table(src_id = graph_output$op.id, src_channel = graph_output$channel.name,
      dst_id = "__terminal__", dst_channel = graph_output$name))

  # add new column to store content that is sent along an edge
  edges$payload = list()
  edges[get("src_id") == "__initial__", "payload" := list(list(input))]

  # get the topo-sorted pipeop ids
  ids = self$ids(sorted = TRUE) # won't contain __initial__  or __terminal__ which are only in our local copy
  edges$fork = FALSE

  # walk over ids, building each operator
  for (id in ids) {
    op = self$pipeops[[id]]
    input_tbl = edges[get("dst_id") == id, list(name = get("dst_channel"), payload = get("payload"))][op$input$name, , on = "name"]
    edges[get("dst_id") == id, "payload" := list(list(NULL))]
    input = input_tbl$payload
    names(input) = input_tbl$name

    lg$debug("Running PipeOp '%s$%s()'", id, fun, pipeop = op, input = input)

    # TODO: maybe generalize this with reflections
    if (test_r6(op, "TorchOpFork")) {
      edges[get("src_id") == id, fork := TRUE]
      layer = NULL
      output = input
    } else {
      layer = op$build(input, task, y)
      output = with_no_grad(
        invoke(layer, .args = input)
      )
    }

    layers[[id]] = layer
    edges[list(id, op$output$name), "payload" := list(output), on = c("src_id", "src_channel")]

  }

  # get payload of edges that go to terminal node.
  # can't use 'dst_id == "__terminal__", because output channel names for Graphs may be duplicated.
  output_tbl = edges[list(graph_output$op.id, graph_output$channel.name),
    c("dst_channel", "payload"), on = c("src_id", "src_channel")]
  output = output_tbl$payload
  names(output) = output_tbl$dst_channel
  filter_noop(output)
  edges$payload = NULL


  return(list(layers = layers, output = output, edges = edges))
}

#' This directly connects the parents of a fork to its branches
simplify_graph = function(edges) {
  edges = copy(edges)
  fork_ids = unique(edges[get("fork")]$src_id)
  for (fork_id in fork_ids) {
    # a  fork looks like:
    #                  --> linear1
    # linear0 --> fork
    #                  --> linear2
    # we want to connect parent(fork) to linear1 and linear2, the parent is:
    # parent(fork) = linear0
    # -----------------------------
    parent = edges[fork_id, "src_id", on = "dst_id"][[1L]]
    # we connect all the branches of the fork (linear1 and linear2) to the parent(fork) = linear0
    #         --> linear1
    # linear0
    #         --> linear2
    edges[fork_id, src_id := parent, on = "src_id"]
  }
  # we remove:
  # linear0 --> fork
  # as these connections are not needed but still in the table
  edges = edges[!fork_ids, on = "dst_id"]
  return(edges)
}
