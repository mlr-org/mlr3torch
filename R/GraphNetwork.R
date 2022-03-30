nn_graph_network = nn_module(
  "graph_network",
  initialize = function(edges, layers) {
    # TODO: maybe check for topological sort?
    edges = copy(edges)
    edges = edges[order(dst_channel), list(src_channel, dst_id, dst_channel, payload), by = src_id]
    edges$input = list(list())
    self$.edges = edges
    self$.layers = layers
    self$.ids = names(layers)
  },
  forward = function(input) {
    network_forward(self$.layers, self$.edges, input)

  }
)

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

network_forward = function(layers, edges, input, ids) {
  input = list(input = input)
  for (i in seq_along(layers)) {
    # get the input for the layer
    # apply the layer
    output = do.call(layers[[i]], args = input)
    # safe the output to the inputs of the next layers
    edges[ids[[i]], input := output, on = "src_id"]
    # prepare the input for the next round
    if (i <= length(layers) - 1L) { # TODO: make this prettier
      input = edges[ids[[i + 1L]], list(dst_channel = dst_channel, input = input), on = "dst_id"]
      input = set_names(
        input$input,
        input$dst_channel
      )
    }
  }
  return(output)
}
