#' @title Graph Network
#' @description
#'
#' Represents a NN using a [`Graph`] that contains [`PipeOpModule`]s.
#'
# '@export
nn_graph = nn_module(
  "nn_graph",
  #' @param model_descriptor ([`ModelDescriptor`])\cr
  #'   Model Descriptor. `.pointer` is ignored, instead `output_pointer` values are used.
  #' @param output_pointers (`list` of `character`)\cr
  #'   Collection of `.pointer`s that indicate what part of the `model_descriptor$graph` is being used for output.
  #'   Entries have the format of `ModelDescriptor$.pointer`.
  #' @param list_output (`logical(1)`)\cr
  #'   Whether output should be a list of tensors. If `FALSE`, then `length(output_pointers)` must be 1.
  initialize = function(model_descriptor, output_pointers, list_output = FALSE) {
    assert_class(model_descriptor, "ModelDescriptor")
    self$list_output = assert_flag(list_output)
    graph = model_descriptor$graph
    self$graph = graph

    self$graph_input_name = graph$input$name  # cache this, it is expensive

    # all graph inputs have an entry in self$shapes_in
    # ModelDescriptor allows Graph to grow by-reference and therefore may have
    # an incomplete $ingress-slot. However, by the time we create an nn_graph,
    # the `graph` must be final, so $ingress must be complete.
    assert_names(names(model_descriptor$ingress), permutation.of = self$graph_input_name)

    self$shapes_in = map(model_descriptor$ingress, "shape")

    assert_list(output_pointers, types = "character", len = if (!list_output) 1)

    self$output_map = character(0)

    graph_output = graph$output  # cache this, it is expensive
    self$output_map = map_chr(output_pointers, function(op) {
      assert_character(op, len = 2, any.missing = FALSE)
      # output pointer referring to graph channel.
      op_name = paste(op, collapse = ".")
      # is this a terminal channel?
      # note we don't just rely on matching op_canonical with output channel name, since
      # pipeop 'a.b' with channel 'c' would produce the same name as pipeop 'a' with channel 'b.c'.
      channel_match = graph_output[as.list(op), on = c("op.id", "channel.name"), nomatch = NULL]
      if (!nrow(channel)) {
        # The indicated channel is not terminal. May happen if output of operation1 gets routed
        # to operation2 and *also* to output: the graph doesn't know that operation1's result should
        # be an output as well --> we add a nop-pipeop to create a terminal channel
        nopid = unique_id(paste(c("output", op), collapse = "_"), names(graph$pipeops))
        graph$add_pipeop(po("nop", id = nopid))
        graph$add_edge(src_id = op[[1]], src_channel = op[[2]], dst_id = nopid)
        op_name = paste0(nopid, ".output")
      }
      op_name
    })

    # the following is necessary to make torch aware of all the included parameters
    mops = Filter(function(x) inherits(x, "PipeOpModule"), graph$pipeops)
    self$modules = nn_module_list(map(mops, "module"))
  },
  forward = function(...) {
    inputs = argument_matcher(names(self$shapes_in))(...)

    inputs = unname(inputs[self$input_map[self$graph_input_name]])

    outputs = self$graph$train(inputs, single_input = FALSE)

    outputs = outputs[self$output_map]

    if (!self$list_output) outputs = outputs[[1]]

    outputs
  }
)

# a function that has argument names 'names' and returns its arguments as a named list.
# used to simulate argument matching for `...`-functions.
# example:
# f = argument_matcher(c("a", "b", "c"))
# f(1, 2, 3) --> list(a = 1, b = 2, c = 3)
# f(1, 2, a = 3) --> list(a = 3, b = 1, c = 2)
# usecase:
# ff = function(...) {
#   l <- argument_matcher(c("a", "b", "c"))(...)
#   l$a + l$b
# }
# # behaves like
# ff(a, b, c) a + b
# (Except in the awquard case of missing args)
argument_matcher = function(names) {
  local(as.function(c(named_list(names, substitute()), quote(as.list(environment())))), envir = topenv())
}

# make unique ID that starts with 'newid' and appends _1, _2, etc. to avoid collisions
unique_id = function(newid, existing_ids) {
  proposal = newid
  i = 1
  while (proposal %in% existing_ids) {
    proposal = sprintf("%s_%s", newid, i)
    i = i + 1
  }
  proposal
}
