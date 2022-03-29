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
  public = list(
    #' @field ptr (`character(1)`) Points to the last node that was added to the Architecture
    ptr = NULL,
    #' @description Initializes an object of class Architecture.
    initialize = function() {
      # This is an environment so we can have shallow copies with different ptrs that build up
      # the same architecture (relevant when forking)
      private$storage = new.env(parent = emptyenv())
      private$storage = Graph$new()
      private$storage$nodes = list()
      private$storage$edges = data.table(
        src_id = character(0),
        src_channel = character(0),
        dst_id = character(0),
        dst_channel = character(0)
      )
      data.table(
        id = character(0),
        parents = list(),
        children = list()
      )
      private$storage$builder = list()
      self$ptr = character(0)
    },

    #' @description add_node
    #' @param x (mlr3torch::TorchOp)
    #' @param parents
    add_node = function(x) {
      id = x$id
      builder = x$build
      assert_true(id %nin% self$ids())
      new_node = data.table(
        id = id,
        parents = list(self$ptr),
        children = list(list())
      )
      private$storage$nodes = rbind(private$storage$nodes, new_node)
      # now we update the children
      for (parent in parents) { # when adding a new node, we have to update some parent's kids
        old_kids =
          i = which(self$ids() == parent)
        new_kids = c(self$nodes[parent, "children", on = "id"][[1L]][[1L]], id)
        set(private$storage$nodes, i, "children", new_kids)
      }
      private$storage$builder[[id]] = builder
      self$ptr = id
      invisible(self)
    },
    add_edge = function() {
      #' @descripion reduces an Architecture for the provided task, using the input as the input
      #' to the first layer
      reduce = function(task, input = NULL) {
        architecture_reduce(self, task, input)
      }
    },
    #' @description Returns the ids of the nodes
    ids = function(sorted = FALSE) {
      if (sorted) {
        ids = topo_sort(private$storage$nodes[, c("id", "parents")])[["id"]]
        return(ids)
      }
      private$storage$nodes[["id"]]
    },
    fork = function(name) {
      architecture = self$copy(deep = FALSE)
      architecture$ptr = name
      return(architecture)
    },
    #' Returns list of parents for the id
    # TODO: rename idx to id but data.table is so fucking annoying
    parents = function(id) {
      private$storage$nodes[id, "parents", on = "id"][[1L]]
    },
    builder = function(id) {
      private$storage$nodes[id, "builder", on = "id"][[1L]][[1L]]
    }
  ),
  active = list(
    #' @field edges
    edges = function(rhs) {
      assert_ro_binding(rhs)
      private$storage$edges
    },
    #' @field builders
    builders = function(rhs) {
      assert_ro_binding(rhs)
      private$storage$builder
    },
    #' @field nodes
    nodes = function(rhs) {
      assert_ro_binding(rhs)
      private$storage$nodes
    }
  ),
  private = list(
    storage = NULL
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
get_instance = function(task) {
  #' TODO: Change this to "meta"
  data_loader = make_dataloader(task, 1, "cpu")
  instance = data_loader$.iter()$.next()
  return(instance)
}

architecture_reduce = function(architecture, task, input = NULL) {
  # 1. G
  ids = architecture$ids(sorted = TRUE)
  builders = architecture$builders
  nodes = architecture$nodes
  nodes[["input"]] = replicate(nrow(nodes), list())
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
  # the orphan is the start of the neural network because (s)he has no parents.
  # There can currently be only one orphan
  orphan = get_orphan(nodes)
  i = which(nodes$id == orphan)
  set(nodes, i, "input", input)
  layers = list()

  for (id in ids) {
    parents = unlist(nodes[id, "parents", on = "id"][[1L]])
    # 3 cases:
    # 1. Tokenizer: gets multiple inputs from the dataloader
    # 2. Standatrd layers: have one input and one output
    # 3. Fork: Have one input and many outputs
    # 3. Merge: Have many inputs and one output
    builder = builders[[id]]
    # TODO: assignment of parameters?
    # two cases: builder has 1 argument --> easy
    # builder has multiple arguments: if ... / order irrelevant --> easy,
    # else care about correct assignment
    args = c(input, )
    args = c(nodes[id, "input", on = "id"][[1L]][[1L]], task = task, y = y)
    layer = invoke(builder, .args = args)
    # layer_args = formalArgs(layer$forward)
    output = with_no_grad(
      layer(input)
    )
    for (child in children) {
      # no we put the input for the children where it belongs
      i = which(nodes$id == child)
      #
      set(nodes, i, "input") = c(nodes[id, "input", on = "id"][[1L]][[1L]], output)
    }
    i = which(nodes$id == id)
    # free the tensor (in case this was the last node for which it was needed)
    set(nodes, i, "input", NULL)
    layers[[id]] = layer
  }
}


make_network = function(layers, nodes, orphans) {
  nodes = add_children(nodes)
  nodes$builder = NULL
  nn_module("network",
    initialize = function(layers, nodes, orphans) {
      # at this point the layers must already be ordered
      assert_true(length(orphans) == 1L)
      imap(
        layers,
        function(value, name) {
          self[[name]] = value
        }
      )
      private$.layers = layers
      private$.ids = names(layers)
      private$.orphan = orphans
      private$.nodes = nodes
    },
    forward = function(input) {
      network_forward(input, private$.layers, private$.nodes, private$.orphan)
    },
    private = list(
      .ids = NULL,
      .orphan = NULL,
      .nodes = NULL,
      .layers = NULL
    )
  )$new(layers, nodes, orphans)
}

# TODO: Refactor this so all the information that is needed for each forward pass has to be
# calculated only once
network_forward = function(input, layers, nodes, orphan) {
  ids = names(layers)
  # 1. We get the nodes that directly get the input
  # note that input here is a list of x_num, x_cat
  # These are the orphans
  # 2. Then we put the input in the corresponding input layer
  # 3. We loop over the ids and apply the layer, after a layer is applied we append the
  # input to the input list of the node
  nodes[["input"]] = replicate(nrow(nodes), list())
  nodes[id == orphan, "input"][[1L]][[1L]] = input

  for (i in seq_along(layers)) {
    layer = layers[[i]]
    idx = ids[[i]]

    input = nodes[idx, "input", on = "id"][[1L]][[1L]]
    output = layer(input)
    children = unlist(nodes[idx, children, on = "id"])
    # childrens = unlist(nodes[id == idx, children][[1L]])
    for (child in children) {
      # TODO: differentiate between mergers and normal layers
      if (length(nodes[child, parents, on = "id"][[1L]]) <= 1L) {
        # nodes[child, input, on = "id"] = output
        nodes[id == child, "input"][[1L]][[1L]] = output
      } else {
        nodes[id == child, input][[1L]] = c(
          nodes[id == child, input][[1L]],
          output
        )
      }
      # we can empty the tensor cache for this node to free memory for the gc
      nodes[id == idx, "input"][[1L]]
    }
  }
  return(output)
}

# # this loops over the nodes of an architecture and appends the column children
# add_children = function(nodes) {
#   nodes$children = replicate(nrow(nodes), list())
#   for (i in seq_len(nrow(nodes))) {
#     parents = nodes[i, "parents"][[1L]]
#     id = nodes[i, "id"][[1L]]
#     if (length(parents[[1L]])) {
#       for (parent in parents) {
#         children = nodes[parent, "children", on = "id"][[1L]][[1L]]
#         # TODO use :=
#         nodes[id == parent, "children"][[1L]][[1L]] = c(children, list(id))
#       }
#     }
#   }
#   return(nodes)
# }
