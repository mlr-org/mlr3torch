# TODO: Add debug option to the forward functoin (keeps tensors in the data.table while the other
# one frees them)
nn_graph = nn_module(
  "nn_graph",
  initialize = function() {
    # TODO: maybe check for topological sort?
    private$.edges = setDT(named_list(c("src_id", "dst_id", "src_channel", "dst_channel"), character(0)))
    private$ids = character(0)
    private$.outputs = list()
    private$.inputs = list()
    # edges = edges[order(dst_channel), list(src_channel, dst_id, dst_channel, payload), by = src_id]
  },
  forward = function(input) {
    edges = private$.edges
    # loads the input into all channels
    edges["__initial__", payload := list(list(..input)), on = "src_id"]
    for (id in private$.ids) {
      layer = self[[id]]
      input_name = private$.inputs[[id]]
      output_name = private$.outputs[[id]]
      # every input channel of the current layer is the dst_channel exactly once
      # Not really sure why we sort according to the name afterwards
      input_tbl = edges[get("dst_id") == id, list(name = get("dst_channel"), payload = get("payload"))][input_name, , on = "name"]
      # we clear the input of the current layer
      edges[get("dst_id") == id, "payload" := list(list(NULL))]
      # and generate the input for the current layer
      input = input_tbl$payload
      names(input) = input_tbl$name

      lg$debug("Running PipeOp '%s$%s()'", id, fun, pipeop = op, input = input)

      # this now either outputs a tensor or a named list of tensors
      output = do.call(layer, args = input)
      if (inherits(output, "torch_tensor")) {

        output = set_names(list(output), output_name)
      }

      edges[list(id, output_name), "payload" := list(..output[get("src_channel")]), on = c("src_id", "src_channel")]
    }
    return(output[[1L]])
  },
  add_edge = function(src_id, dst_id, src_channel, dst_channel) {
    row = data.table(src_id, src_channel, dst_id, dst_channel)
    private$.edges = rbind(private$.edges, row)
  },
  add_layer = function(id, layer, inputs, outputs) {
    self[[id]] = layer
    private$.ids = c(private$.ids, id)
    private$.inputs[[id]] = inputs
    private$.outputs[[id]] = outputs
  },
  edges = function(rhs) {
    assert_ro_biding(rhs)
    private$.edges
  }
)
