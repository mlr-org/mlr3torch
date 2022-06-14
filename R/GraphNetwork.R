# TODO: Add debug option to the forward functoin (keeps tensors in the data.table while the other
# one frees them)
nn_graph = nn_module(
  "nn_Graph",
  initialize = function() {
    # TODO: maybe check for topological sort?
    private$.edges = setDT(named_list(c("src_id", "dst_id", "src_channel", "dst_channel"), character(0)))
    private$ids = character(0)
    # edges = edges[order(dst_channel), list(src_channel, dst_id, dst_channel, payload), by = src_id]
  },
  forward = function(input) {
    edges = private$.edges
    edges["__initial__", input := list(list(..input)), on = "src_id"]
    for (id in private$.ids) {
      input = edges[id, list(dst_channel = get("dst_channel"), input = input), on = "dst_id"]
      input = set_names(input$input, input$dst_channel)
      output = do.call(self[[id]], args = input)
      edges[id, input := output, on = "src_id"]
    }

    return(output)
  },
  add_edge = function(src_id, dst_id, src_channel, dst_channel) {
    row = data.table(src_id, src_channel, dst_id, dst_channel)
    private$.edges = rbind(private$.edges, row)
  },
  add_layer = function(id, layer) {
    self[[id]] = layer
    private$.ids = c(private$.ids, id)
  },
  edges = function(rhs) {
    assert_ro_biding(rhs)
    private$.edges
  }
)


