#' @title Graph Network
#'
#' @description
#' Represents a neural network using a [`Graph`][mlr3pipelines::Graph] that usually costains mostly [`PipeOpModule`]s.
#'
#' @param graph ([`Graph`][mlr3pipelines::Graph])\cr
#'   The [`Graph`][mlr3pipelines::Graph] to wrap. Is **not** cloned.
#' @param shapes_in (named `integer`)\cr
#'   Shape info of tensors that go into `graph`. Names must be `graph$input$name`, possibly in different order.
#' @param output_map (`character`)\cr
#'   Which of `graph`'s outputs to use. Must be a subset of `graph$output$name`.
#' @param list_output (`logical(1)`)\cr
#'   Whether output should be a list of tensors. If `FALSE` (default), then `length(output_map)` must be 1.
#'
#' @return [`nn_graph`]
#' @family Graph Network
#' @export
#' @examplesIf torch::torch_is_installed()
#' graph = mlr3pipelines::Graph$new()
#' graph$add_pipeop(po("module_1", module = nn_linear(10, 20)), clone = FALSE)
#' graph$add_pipeop(po("module_2", module = nn_relu()), clone = FALSE)
#' graph$add_pipeop(po("module_3", module = nn_linear(20, 1)), clone = FALSE)
#' graph$add_edge("module_1", "module_2")
#' graph$add_edge("module_2", "module_3")
#'
#' network = nn_graph(graph, shapes_in = list(module_1.input = c(NA, 10)))
#'
#' x = torch_randn(16, 10)
#'
#' network(module_1.input = x)
nn_graph = nn_module(
  "nn_graph",
  initialize = function(graph, shapes_in, output_map = graph$output$name, list_output = FALSE) {
    self$graph = as_graph(graph, clone = FALSE)
    self$graph_input_name = graph$input$name  # cache this, it is expensive

    # we do NOT verify the input and type of the graph to be `"torch_tensor"`.
    # The reason for this is that the graph, when constructed with the PipeOpTorch Machinery, contains PipeOpNOPs,
    # which have input and output type *.

    self$list_output = assert_flag(list_output)
    assert_names(names(shapes_in), permutation.of = self$graph_input_name)
    self$shapes_in = assert_list(shapes_in, types = "integerish")
    self$output_map = assert_subset(output_map, self$graph$output$name)
    if (!list_output && length(output_map) != 1) {
      stopf("If list_output is FALSE, output_map must have length 1.")
    }

    self$argument_matcher = argument_matcher(names(self$shapes_in))

    # the following is necessary to make torch aware of all the included parameters
    # (some operators in the graph could be different from PipeOpModule, e.g. PipeOpBranch or PipeOpNOP
    mops =  Filter(function(x) inherits(x, "PipeOpModule"), graph$pipeops)
    # dont use modules name as it is reserved
    self$module_list = nn_module_list(keep(map(mops, "module"), is, "nn_module"))
  },
  forward = function(...) {
    # this ensures that the arguments are passed in the correct order
    inputs = self$argument_matcher(...)

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
  },
  private = list(
    finalize_deep_clone = function() {
      # here we need to clone the graph without cloning
      # this assumes that the order of the PipeOps in self$graph does not change
      # the problem is that nn_module_list() does not preserve the names
      module_ids = discard(.p = is.null, map(self$graph$pipeops, function(po) {
        if (!test_class(po, "PipeOpModule")) return(NULL)
        if (!test_class(po$module, "nn_module")) return(NULL)
        po$module = NULL
        po$id
      }))
      self$graph = self$graph$clone(deep = TRUE)
      for (i in seq_along(module_ids)) {
        # the first module is self
        self$graph$pipeops[[module_ids[[i]]]]$module = self$module_list$modules[[i + 1]]
        i = i + 1
      }
      invisible(self)
    }
  )
)


#' @title Create a nn_graph from ModelDescriptor
#'
#' @description
#' Creates the [`nn_graph`] from a [`ModelDescriptor`]. Mostly for internal use, since the [`ModelDescriptor`] is in
#' most circumstances harder to use than just creating [`nn_graph`] directly.
#'
#' @param model_descriptor ([`ModelDescriptor`])\cr
#'   Model Descriptor. `pointer` is ignored, instead `output_pointer` values are used. `$graph` member is
#'   modified by-reference.
#' @param output_pointers (`list` of `character`)\cr
#'   Collection of `pointer`s that indicate what part of the `model_descriptor$graph` is being used for output.
#'   Entries have the format of `ModelDescriptor$pointer`.
#' @param list_output (`logical(1)`)\cr
#'   Whether output should be a list of tensors. If `FALSE`, then `length(output_pointers)` must be 1.
#'
#' @return [`nn_graph`]
#' @family Graph Network
#' @export
model_descriptor_to_module = function(model_descriptor, output_pointers = NULL, list_output = FALSE) {
  assert_class(model_descriptor, "ModelDescriptor")
  assert_flag(list_output)
  assert_list(output_pointers, types = "character", len = if (!list_output) 1, null.ok = TRUE)
  output_pointers = output_pointers %??% list(model_descriptor$pointer)

  # all graph inputs have an entry in self$shapes_in
  # ModelDescriptor allows Graph to grow by-reference and therefore may have
  # an incomplete $ingress-slot. However, by the time we create an nn_graph,
  # the `graph` must be final, so $ingress must be complete.
  shapes_in = map(model_descriptor$ingress, "shape")

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
      # note that op = c(id, channel)
      nopid = unique_id(paste(c("output", op[1L]), collapse = "_"), names(graph$pipeops))

      graph$add_pipeop(po("nop", id = nopid))
      graph$add_edge(src_id = op[[1]], src_channel = op[[2]], dst_id = nopid)
      op_name = paste0(nopid, ".output")
    }
    op_name
  })

  nn_graph(graph, shapes_in, output_map, list_output = list_output)
}

#' @title Create a Torch Learner from a ModelDescriptor
#'
#' @description
#' First a [`nn_graph`] is created using [`model_descriptor_to_module`] and then a learner is created from this
#' module and the remaining information from the model descriptor, which must include the optimizer and loss function
#' and optionally callbacks.
#'
#' @param model_descriptor ([`ModelDescriptor`])\cr
#'   The model descriptor.
#' @return [`Learner`][mlr3::Learner]
#' @family Graph Network
#' @export
model_descriptor_to_learner = function(model_descriptor) {
  optimizer = as_torch_optimizer(model_descriptor$optimizer)
  loss = as_torch_loss(model_descriptor$loss)
  callbacks = as_torch_callbacks(model_descriptor$callbacks)
  task_type = model_descriptor$task$task_type

  network = model_descriptor_to_module(
    model_descriptor = model_descriptor,
    output_pointers = list(model_descriptor$pointer),
    list_output = FALSE
  )
  ingress_tokens = model_descriptor$ingress
  network$reset_parameters()

  learner = LearnerTorchModel$new(
    task_type = task_type,
    network = network,
    properties = NULL,
    ingress_tokens = ingress_tokens,
    optimizer = optimizer,
    loss = loss,
    callbacks = callbacks,
    # The packages of the loss, optimizer and callbacks are added anyway (?)
    packages = model_descriptor$graph$packages
  )

  return(learner)
}

# a function that has argument names 'names' and returns its arguments as a named list.
# used to simulate argument matching for `...`-functions.
# example:
# f = argument_matcher(c("a", "b", "c"))
# f(1, 2, 3) --> list(a = 1, b = 2, c = 3)
# f(1, 2, a = 3) --> list(a = 3, b = 1, c = 2)
# usecase:
# ff = function(...) {
#   l = argument_matcher(c("a", "b", "c"))(...)
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
