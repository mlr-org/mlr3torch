#' @title Architecture
#' @description This is the container object that stores the information from the TorchOps.
#' The nodes and edges are stored in an environment, because due to the TorchOpParallel [name?]
#' operator there can be multiple shallow copies of this architecture that are identical except
#' for the ptr field that is the current output of the architecture. Because of this they
#' are stored in environments that can be simultaneously modified through all the shallow copies.
#'
Graphitecture = R6Class("Graphitecture",
  storage = NULL,
  ptr = NULL,
  public = list(
    initialize = function() {
      self$storage = new.env(parent = emptyenv())
      self$storage$nodes = data.table(
        id = character(0),
        parents = list(),
        builders = list()
      )
      self$ptr = "..input"
    },
    add_node = function(id, builder) {
      assert_true(id %nin% self$ids())
      new_node = data.table(
        id = id,
        parents = list("id"),
        builder = list(builder)
      )
      rbind(self$storage$nodes, new_node)
    },
    reduce = function(task) {
      architecture_reduce(self, task)
    },
    #' @description Returns the ids of the nodes
    ids = function(sorted = FALSE) {
      if (sorted) {
        return(mlr3misc::topo_sort(self$storage$nodes))
      }
      self$storage$nodes[["id"]]
    },
    builders = function(sorted) {
      self$storage$builders[[self$ids(sorted)]]
    },
    #' Returns list of parents for the id
    parents = function(id) {
      self$storage$nodes[id == id, "parents", with = FALSE][[1L]]
    }
  ),
  active = list(
    edges = function(rhs) {
      assert_ro_binding(rhs)
      self$storage$edges
    },
    nodes = function(rhs) {
      assert_ro_binding(rhs)
      self$storage$nodes
    },
    ids = function(rhs) {
      assert_ro_binding(rhs)
      self$storage$nodes[["id"]]
    }
  )
)

make_nn_module = function(layers) {
  nn_module(
    initialize = function(layers) {
      for (layer i )
    }
    forwa
  )
}

architecture_reduce = function(architecture, task) {
  ids = architecture$ids(sorted = TRUE)
  # now we add the start node and the end node

  layers = list()
  tensors = list() # here we store the intermediate results (outputs of the nodes)
  # that are needed to build the next layer, the keys are the ids and the values the tensors
  data_loader = make_dataloader(task, 1, "cpu") # TODO: don't build full dataloader here
  tensors[["..input"]] = data_loader$.iter()$.next()

  for (id in ids) {
    # We get the builder function
    builder = architecture$builders[[]]
    #



    layer =
    layers = append(layers)
  }
}

#' This builds a nn_module from the graph architecture
#' Even though the grpah is already sorted once
#'
reduce_graph_architecture = function(arc, task) {
  mlr3misc::topo_sort

}

make_parents = function(edges, nodes) {

}

id         parents
<char>          <list>
1:    no_op          branch
2:      pca          branch
3:    scale          branch
4: unbranch no_op,pca,scale
5:   branch
