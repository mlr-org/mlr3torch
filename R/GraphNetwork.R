# TODO: Add debug option to the forward functoin (keeps tensors in the data.table while the other
# one frees them)
nn_graph = nn_module(
  "nn_Graph",
  initialize = function(edges, layers) {
    # TODO: maybe check for topological sort?
    edges = copy(edges)
    # edges = edges[order(dst_channel), list(src_channel, dst_id, dst_channel, payload), by = src_id]
    imap(
      layers,
      function(value, name) {
        self[[name]] = value
      }
    )
    private$.edges = edges
    private$.edges$input = list(list())
    private$.layers = layers
    private$.ids = names(layers)
  },
  forward = function(input) {
    network_forward(private$.layers, private$.edges, private$.ids, input)
  }
)


network_forward = function(layers, edges, ids, input) {
  edges["__initial__", input := list(list(input)), on = "src_id", with = FALSE]
  for (i in seq_along(layers)) {
    input = edges[ids[[i]], list(dst_channel = get("dst_channel"), input = input), on = "dst_id"]
    input = set_names(
      input$input,
      input$dst_channel
    )
    output = do.call(layers[[i]], args = input)
    edges[ids[[i]], input := output, on = "src_id"]
  }
  return(output)
}
