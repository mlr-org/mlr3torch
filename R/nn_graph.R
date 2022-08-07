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
    self$shapes_in = model_descriptor$shapes_in
    self$input_map = model_descriptor$input_map

    assert_list(output_pointers, types = "character", len = if (!list_output) 1)
    self$output_map = character(0)
    outnames = graph$output$name

    for (idx in seq_along(output_pointers)) {
      op = output_pointers[[idx]]
      assert_character(op, min.len = 1, max.len = 2, any.missing = FALSE)
      if (length() == 1) {
        # output pointer referring to input shape
        # --> we add a PipeOpNop that just pipes the input through.
        nopid = unique_id(paste(c("nop", op), collapse = "_"), names(graph$pipeops))
        graph$add_pipeop(po("nop", id = nopid))
        self$input_map[[paste0(nopid, ".input")]] = op
        self$output_map[[idx]] = paste0(nopid, ".output")
      } else {
        # output pointer referring to graph channel.
        op_canonical = paste(op, collapse = ".")
        # is this a terminal channel?
        if (op_canonical %nin% outnames) {
          # no --> we add a nop-pipeop to create a terminal channel
          nopid = unique_id(paste(c("output", op), collapse = "_"), names(graph$pipeops))
          graph$add_pipeop(po("nop", id = nopid))
          graph$add_edge(src_id = op[[1]], src_channel = op[[2]], dst_id = nopid)
          op_canonical = paste0(nopid, ".output")
        }
        self$output_map[[idx]] = op_canonical
      }
    }

    self$graph_input_name = graph$input$name  # cache this, it is expensive

    # sanity checks all graph inputs have an entry in self$input_map
    assert_names(names(self$input_map), permutation.of = self$graph_input_name)

    pos = Filter(function(x) inherits(x, "PipeOpModule"), graph$pipeops)
    self$modules = nn_module_list(map(pos, "module"))  # this is necessary to make torch aware of all the included parameters
    self$graph = graph
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
