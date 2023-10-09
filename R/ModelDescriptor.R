#' @title Represent a Model with Meta-Info
#'
#' @description
#' Represents a *model*; possibly a complete model, possibly one in the process of being built up.
#'
#' This model takes input tensors of shapes `shapes_in` and
#' pipes them through `graph`. Input shapes get mapped to input channels of `graph`.
#' Output shapes are named by the output channels of `graph`; it is also possible
#' to represent no-ops on tensors, in which case names of input and output should be identical.
#'
#' `ModelDescriptor` objects typically represent partial models being built up, in which case the `.pointer` slot
#' indicates a specific point in the graph that produces a tensor of shape `.pointer_shape`, on which the graph should
#' be extended.
#' It is allowed for the `graph` in this structure to be modified by-reference in different parts of the code.
#' However, these modifications may never add edges with elements of the `Graph` as destination. In particular, no
#' element of `graph$input` may be removed by reference, e.g. by adding an edge to the `Graph` that has the input
#' channel of a `PipeOp` that was previously without parent as its destination.
#'
#' In most cases it is better to create a specific `ModelDescriptor` by training a [`Graph`] consisting (mostly) of
#' operators [`PipeOpTorchIngress`], [`PipeOpTorch`], [`PipeOpTorchLoss`], [`PipeOpTorchOptimizer`], and
#' [`PipeOpTorchCallbacks`].
#'
#' A `ModelDescriptor` can be converted to a [`nn_graph`] via [`model_descriptor_to_module`].
#'
#' @param graph ([`Graph`][mlr3pipelines::Graph])\cr
#'   `Graph` of [`PipeOpModule`] and [`PipeOpNOP`] operators.
#' @param ingress (uniquely named `list` of `TorchIngressToken`)\cr
#'   List of inputs that go into `graph`. Names of this must be a subset of `graph$input$name`.
#' @param task ([`Task`][mlr3::Task])\cr
#'   (Training)-Task for which the model is being built. May be necessary for for some aspects of what loss to use etc.
#' @param optimizer ([`TorchOptimizer`] | `NULL`)\cr
#'   Additional info: what optimizer to use.
#' @param loss ([`TorchLoss`] | `NULL`)\cr
#'   Additional info: what loss to use.
#' @param callbacks (A `list` of [`CallbackSet`] or `NULL`)\cr
#'   Additional info: what callbacks to use.
#' @param .pointer (`character(2)` | `NULL`)\cr
#'   Indicating an element on which a model is. Points to an output channel within `graph`:
#'   Element 1 is the `PipeOp`'s id and element 2 is that `PipeOp`'s output channel.
#' @param .pointer_shape (`integer` | `NULL`)\cr
#'   Shape of the output indicated by `.pointer`
#' @param .loader_outputs (`list()` of `character(2)`)\cr
#'   A list of name-channel pairs that are the outputs of the data loader that is returned from the parallel workers
#'   to the main workers.
#'
#'
#' @family Model Configuration
#' @family Graph Network
#' @return (`ModelDescriptor`)
#' @export
ModelDescriptor = function(graph, ingress, task, optimizer = NULL, loss = NULL, callbacks = NULL, .pointer = NULL,
  .pointer_shape = NULL, .loader_outputs = NULL) {
  assert_r6(graph, "Graph")
  innames = graph$input$name  # graph$input$name access is slow

  assert_list(ingress, min.len = 1, types = "TorchIngressToken", names = "unique", any.missing = FALSE)

  # conditions on ingress: maps shapes_in to graph$input$name
  assert_names(names(ingress), subset.of = innames)

  assert_list(.loader_outputs, null.ok = TRUE)

  assert_r6(task, "Task")

  assert_r6(optimizer, "TorchOptimizer", null.ok = TRUE)
  assert_r6(loss, "TorchLoss", null.ok = TRUE)
  if (!is.null(loss)) {
    assert_choice(task$task_type, loss$task_types)
  }
  callbacks = as_torch_callbacks(callbacks)
  callbacks = set_names(callbacks, assert_names(ids(callbacks), type = "unique"))

  if (!is.null(.pointer)) {
    assert_integerish(.pointer_shape)
    assert_choice(.pointer[[1]], names(graph$pipeops))
    assert_choice(.pointer[[2]], graph$pipeops[[.pointer[[1]]]]$output$name)
  }

  structure(list(
    graph = graph,
    ingress = ingress,
    task = task,
    optimizer = optimizer,
    loss = loss,
    callbacks = callbacks,
    .pointer = .pointer,
    .pointer_shape = .pointer_shape,
    .loader_outputs = .loader_outputs
  ), class = "ModelDescriptor")
}

#' @export
print.ModelDescriptor = function(x, ...) {
  shape_to_str = function(x) {
    shapedescs = map_chr(x, function(y) paste0("(", paste(y, collapse = ",", recycle0 = TRUE), ")"))
    paste0("[",  paste(shapedescs, collapse = ";", recycle0 = TRUE), "]")
  }

  ingress_shapes = imap(x$ingress, function(x, nm) {
    paste0(nm, ": ", shape_to_str(list(x$shape)))
  })

  loader_outputs = paste(map_chr(x$.loader_outputs, function(x) paste0(x[1], ".", x[2])))

  catn(sprintf("<ModelDescriptor: %d ops>", length(x$graph$pipeops)))
  catn(str_indent("* Ingress: ", ingress_shapes))
  catn(str_indent("* Task: ", paste0(x$task$id, " [", x$task$task_type, "]")))
  catn(str_indent("* Callbacks: ", if (!is.null(x$callbacks) && length(x$callbacks)) as_short_string(map_chr(x$callbacks, "label"), 100L) else "N/A")) # nolint
  catn(str_indent("* Optimizer: ", if (!is.null(x$optimizer)) as_short_string(x$optimizer$label) else "N/A"))
  catn(str_indent("* Loss: ", if (!is.null(x$loss)) as_short_string(x$loss$label) else "N/A"))
  catn(str_indent("* .pointer: ", if (is.null(x$.pointer)) "" else { # nolint
    sprintf("\n%s %s", paste(x$.pointer, collapse = "."), shape_to_str(list(x$.pointer_shape)))
  }))
  catn(str_indent("* .loader_outputs: ", loader_outputs))
}

#' @title Union of ModelDescriptors
#'
#' @description
#' This is a mostly internal function that is used in [`PipeOpTorch`]s with multiple input channels.
#'
#' It creates the union of multiple [`ModelDescriptor`]s:
#'
#' * `graph`s are combinded (if they are not identical to begin with). The first entry's `graph` is modified by
#'    reference.
#' * `PipeOp`s with the same ID must be identical. No new input edges may be added to `PipeOp`s.
#' * Drops `.pointer` / `.pointer_shape` entries.
#' * The new task is the [feature union][PipeOpFeatureUnion] of the two incoming tasks.
#' * The `optimizer` and `loss` of both [`ModelDescriptor`]s must be identical.
#' * Ingress tokens and callbacks are merged, where objects with the same `"id"` must be identical.
#'
#' @details
#' The requirement that no new input edgedes may be added to `PipeOp`s  is not theoretically necessary, but since
#' we assume that ModelDescriptor is being built from beginning to end (i.e. `PipeOp`s never get new ancestors) we
#' can make this assumption and simplify things. Otherwise we'd need to treat "..."-inputs special.)
#'
#' @param md1 (`ModelDescriptor`)
#'   The first [`ModelDescriptor`].
#' @param md2 (`ModelDescriptor`)
#'   The second [`ModelDescriptor`].
#' @return [`ModelDescriptor`]
#' @family Graph Network
#' @family Model Configuration
#' @export
model_descriptor_union = function(md1, md2) {
  assert_class(md1, "ModelDescriptor")
  assert_class(md2, "ModelDescriptor")

  graph = merge_graphs(md1$graph, md2$graph)

  # merge tasks: this is pretty much exactly what POFU does, so we use it in the non-trivial case.
  if (identical(md1$task, md2$task)) {
    task = md1$task
  } else {
    task = PipeOpFeatureUnion$new()$train(list(md1$task, md2$task))[[1]]
  }

  ModelDescriptor(
    graph = graph,
    ingress = merge_assert_unique(md1$ingress, md2$ingress, .var.name = "ingress tokens"),
    task = task,
    optimizer = coalesce_assert_id(md1$optimizer, md2$optimizer, .var.name = "optimizer"),
    loss = coalesce_assert_id(md1$loss, md2$loss, .var.name = "loss"),
    callbacks = merge_assert_unique(md1$callbacks, md2$callbacks, .var.name = "callbacks")
  )
}



merge_assert_unique = function(a, b, .var.name) { # nolint
  common_names = intersect(names(a), names(b))
  assert_true(identical(a[common_names], b[common_names]),
    .var.name = sprintf("%s with ID(s) %s are identical.", .var.name, paste0(common_names, collapse = ", ")))
  a[names(b)] = b
  a
}

coalesce_assert_id = function(a, b, .var.name) { # nolint
  if (!is.null(a) && !is.null(b) && !identical(a, b)) {
    stop(sprintf("%s of two ModelDescriptors being merged disagree.", .var.name))
  }
  a %??% b
}
