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
#' `ModelDescriptor` objects typically represent partial models being built up, in which case the `pointer` slot
#' indicates a specific point in the graph that produces a tensor of shape `pointer_shape`, on which the graph should
#' be extended.
#' It is allowed for the `graph` in this structure to be modified by-reference in different parts of the code.
#' However, these modifications may never add edges with elements of the `Graph` as destination. In particular, no
#' element of `graph$input` may be removed by reference, e.g. by adding an edge to the `Graph` that has the input
#' channel of a `PipeOp` that was previously without parent as its destination.
#'
#' In most cases it is better to create a specific `ModelDescriptor` by training a [`Graph`][mlr3pipelines::Graph] consisting (mostly) of
#' operators [`PipeOpTorchIngress`], [`PipeOpTorch`], [`PipeOpTorchLoss`], [`PipeOpTorchOptimizer`], and
#' [`PipeOpTorchCallbacks`].
#'
#' A `ModelDescriptor` can be converted to a [`nn_graph`] via [`model_descriptor_to_module`].
#'
#' @param graph ([`Graph`][mlr3pipelines::Graph])\cr
#'   `Graph` of [`PipeOpModule`] and [`PipeOpNOP`][mlr3pipelines::PipeOpNOP] operators.
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
#' @param pointer (`character(2)` | `NULL`)\cr
#'   Indicating an element on which a model is. Points to an output channel within `graph`:
#'   Element 1 is the `PipeOp`'s id and element 2 is that `PipeOp`'s output channel.
#' @param pointer_shape (`integer` | `NULL`)\cr
#'   Shape of the output indicated by `pointer`.
#'
#' @family Model Configuration
#' @family Graph Network
#' @return (`ModelDescriptor`)
#' @export
ModelDescriptor = function(graph, ingress, task, optimizer = NULL, loss = NULL, callbacks = NULL, pointer = NULL,
  pointer_shape = NULL) {
  assert_r6(graph, "Graph")
  innames = graph$input$name  # graph$input$name access is slow

  assert_list(ingress, min.len = 1, types = "TorchIngressToken", names = "unique", any.missing = FALSE)

  # conditions on ingress: maps shapes_in to graph$input$name
  assert_names(names(ingress), subset.of = innames)

  assert_r6(task, "Task")

  assert_r6(optimizer, "TorchOptimizer", null.ok = TRUE)
  assert_r6(loss, "TorchLoss", null.ok = TRUE)
  if (!is.null(loss)) {
    assert_choice(task$task_type, loss$task_types)
  }
  callbacks = as_torch_callbacks(callbacks)
  callbacks = set_names(callbacks, assert_names(ids(callbacks), type = "unique"))

  if (!is.null(pointer)) {
    pointer_shape = assert_shape(pointer_shape, null_ok = FALSE, unknown_batch = TRUE)
    assert_choice(pointer[[1]], names(graph$pipeops))
    assert_choice(pointer[[2]], graph$pipeops[[pointer[[1]]]]$output$name)
  }

  structure(list(
    graph = graph,
    ingress = ingress,
    task = task,
    optimizer = optimizer,
    loss = loss,
    callbacks = callbacks,
    pointer = pointer,
    pointer_shape = pointer_shape
  ), class = "ModelDescriptor")
}

#' @export
#' @include utils.R
print.ModelDescriptor = function(x, ...) {
  ingress_shapes = imap(x$ingress, function(x, nm) {
    paste0(nm, ": ", shape_to_str(list(x$shape)))
  })

  catn(sprintf("<ModelDescriptor: %d ops>", length(x$graph$pipeops)))
  catn(str_indent("* Ingress: ", ingress_shapes))
  catn(str_indent("* Task: ", paste0(x$task$id, " [", x$task$task_type, "]")))
  catn(str_indent("* Callbacks: ", if (!is.null(x$callbacks) && length(x$callbacks)) as_short_string(map_chr(x$callbacks, "label"), 100L) else "N/A")) # nolint
  catn(str_indent("* Optimizer: ", if (!is.null(x$optimizer)) as_short_string(x$optimizer$label) else "N/A"))
  catn(str_indent("* Loss: ", if (!is.null(x$loss)) as_short_string(x$loss$label) else "N/A"))
  catn(str_indent("* pointer: ", if (is.null(x$pointer)) "" else { # nolint
    sprintf("\n%s %s", paste(x$pointer, collapse = "."), shape_to_str(list(x$pointer_shape)))
  }))
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
#' * Drops `pointer` / `pointer_shape` entries.
#' * The new task is the [feature union][mlr3pipelines::PipeOpFeatureUnion] of the two incoming tasks.
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
  graph = md1$graph

  # if graphs are identical, we don't need to worry about copying stuff
  if (!identical(md1$graph, md2$graph)) {
    # PipeOps that have the same ID that occur in both graphs must be identical.
    common_names = intersect(names(graph$pipeops), names(md2$graph$pipeops))
    if (!identical(graph$pipeops[common_names], md2$graph$pipeops[common_names])) {
      not_identical = map_lgl(common_names, function(name) {
        !identical(graph$pipeops[[name]], md2$graph$pipeops[[name]])
      })
      stopf("Both graphs have PipeOps with ID(s) %s but they are not identical.",
        paste0("'", common_names[not_identical], "'", collapse = ", ")
      )
    }

    # copy all PipeOps that are in md2 but not in md1
    graph$pipeops = c(graph$pipeops, md2$graph$pipeops[setdiff(names(md2$graph$pipeops), common_names)])

    # clear param_set cache
    graph$.__enclos_env__$private$.param_set = NULL

    # edges that are in md2's graph that were not in md1's graph
    new_edges = md2$graph$edges[!graph$edges, on = c("src_id", "src_channel", "dst_id", "dst_channel")]

    # IDs and channel names that get new input edges. These channels must not already have incoming edges in md1.
    new_input_edges = unique(new_edges[, c("dst_id", "dst_channel"), with = FALSE])

    forbidden_edges = graph$edges[new_input_edges, on = c("dst_id", "dst_channel"), nomatch = NULL]

    if (nrow(forbidden_edges)) {
      stopf("PipeOp(s) %s have differing incoming edges in md1 and md2.",
        paste(forbidden_edges$dst_id, collapse = ", "))

    }
    graph$edges = rbind(graph$edges, new_edges)
  }

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
