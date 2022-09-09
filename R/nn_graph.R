

#' @title Create nn_module from ModelDescriptor
#'
#' @description
#'
#' Creates the [`nn_graph`] from a [`ModelDescriptor`]. Mostly for internal use, since the [`ModelDescriptor`] is in most
#' circumstances harder to use than just creating [`nn_graph`] directly.
#'
#' @param model_descriptor ([`ModelDescriptor`])\cr
#'   Model Descriptor. `.pointer` is ignored, instead `output_pointer` values are used. `$graph` member is
#'   modified by-reference.
#' @param output_pointers (`list` of `character`)\cr
#'   Collection of `.pointer`s that indicate what part of the `model_descriptor$graph` is being used for output.
#'   Entries have the format of `ModelDescriptor$.pointer`.
#' @param list_output (`logical(1)`)\cr
#'   Whether output should be a list of tensors. If `FALSE`, then `length(output_pointers)` must be 1.
#' @export
model_descriptor_to_module = function(model_descriptor, output_pointers, list_output = FALSE) {
  assert_class(model_descriptor, "ModelDescriptor")
  assert_flag(list_output)

  # all graph inputs have an entry in self$shapes_in
  # ModelDescriptor allows Graph to grow by-reference and therefore may have
  # an incomplete $ingress-slot. However, by the time we create an nn_graph,
  # the `graph` must be final, so $ingress must be complete.
  shapes_in = map(model_descriptor$ingress, "shape")

  assert_list(output_pointers, types = "character", len = if (!list_output) 1)

  graph = model_descriptor$graph

  graph_output = graph$output  # cache this, it is expensive
  output_map = map_chr(output_pointers, function(op) {
    assert_character(op, len = 2, any.missing = FALSE)
    # output pointer referring to graph channel.
    op_name = paste(op, collapse = ".")
    # is this a terminal channel?
    # note we don't just rely on matching op_canonical with output channel name, since
    # pipeop 'a.b' with channel 'c' would produce the same name as pipeop 'a' with channel 'b.c'.
    channel_match = graph_output[as.list(op), on = c("op.id", "channel.name"), nomatch = NULL]
    if (!nrow(channel_match)) {
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

  nn_graph(graph, shapes_in, output_map, list_output = list_output)
}


#' @title Graph Network
#' @description
#'
#' Represents a NN using a [`Graph`] that contains [`PipeOpModule`]s.
#'
#' @export
nn_graph = nn_module(
  "nn_graph",
  #' @param graph ([`Graph`][mlr3pipelines::Graph])\cr
  #'   The [`Graph`][mlr3pipelines::Graph] to wrap.
  #' @param shapes_in (named `integer`)\cr
  #'   Shape info of tensors that go into `graph`. Names must be `graph$input$name`, possibly in different order.
  #' @param output_map (`character`)\cr
  #'   Which of `graph`'s outputs to use. Must be a subset of `graph$output$name`.
  #' @param list_output (`logical(1)`)\cr
  #'   Whether output should be a list of tensors. If `FALSE`, then `length(output_map)` must be 1.
  initialize = function(graph, shapes_in, output_map = graph$output$name, list_output = FALSE) {

    self$list_output = assert_flag(list_output)
    self$graph = graph

    self$graph_input_name = graph$input$name  # cache this, it is expensive

    self$shapes_in = assert_list(shapes_in, types = "integerish")
    assert_names(names(shapes_in), permutation.of = self$graph_input_name)

    self$output_map = assert_subset(output_map, self$graph$output$name)

    # the following is necessary to make torch aware of all the included parameters
    # (some operators in the graph could be different from PipeOpModule, e.g. PipeOpBranch or PipeOpNOP
    mops = Filter(function(x) inherits(x, "PipeOpModule"), graph$pipeops)
    self$modules = nn_module_list(map(mops, "module"))
  },
  forward = function(...) {
    inputs = argument_matcher(names(self$shapes_in))(...)

    outputs = self$graph$train(unname(inputs), single_input = FALSE)

    outputs = outputs[self$output_map]

    if (!self$list_output) outputs = outputs[[1]]

    outputs
  },
  reset_parameters = function() {
    # recursively call $reset_parameters()
    recursive_reset = function(network) {
      for (child in network$children) {
        if (is.function(child$reset_parameters)) {
          child$reset_parameters()
          next
        }
        Recall(child)
      }
    }

    recursive_reset(self)
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
argument_matcher = function(args) {
  fn = as.function(c(named_list(args, substitute()), quote(as.list(environment()))))
  environment(fn) = topenv()
  fn
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
