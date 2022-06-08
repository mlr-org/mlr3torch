Graphitecture = R6Class("Graphitecture",
  inherit = Graph,
  public = list(
    add_torchop = function(op) {
      assert_true(inherits(op, "TorchOp"))
      super$add_pipeop(op)
    },
    build = function(task, input = NULL) {
      reduction = graphitecture_reduce(self, task, input)
      # edges  simplify_graph(reduction$edges)
      nn_graph$new(reduction$edges, reduction$layers)
    }
  )
)

graphitecture_reduce = function(self, task = NULL, input = NULL) {
  graph_input = self$input
  graph_output = self$output
  assert(
    test_true(nrow(graph_input) == 1L),
    test_true(nrow(graph_output) == 2L)
  )
  layers = list()

  instance = get_batch(task, batch_size = 1L, device = "cpu")
  y = instance[["y"]]
  if (is.null(input)) {
    input = instance$x
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

  # walk over ids, building each operator
  for (id in ids) {
    op = self$pipeops[[id]]
    input_tbl = edges[get("dst_id") == id, list(name = get("dst_channel"), payload = get("payload"))][op$input$name, , on = "name"]
    edges[get("dst_id") == id, "payload" := list(list(NULL))]
    input = input_tbl$payload
    names(input) = input_tbl$name

    lg$debug("Running PipeOp '%s'", id, pipeop = op, input = input)

    c(layers[[id]], output) %<-% op$build(input, task, y)
    # layers[[id]] = layer
    edges[list(id, op$output$name), "payload" := output, on = c("src_id", "src_channel")]

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
