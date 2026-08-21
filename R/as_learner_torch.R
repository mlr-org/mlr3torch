#' @title Convert a Graph to a Torch Learner
#'
#' @description
#' Converts a [`Graph`][mlr3pipelines::Graph] that is built from [`PipeOpTorch`] operators into a
#' [`LearnerTorch`].
#'
#' Calling [`as_learner()`][mlr3::as_learner] on such a graph gives a
#' [`GraphLearner`][mlr3pipelines::GraphLearner], which trains the same network but hides everything
#' that makes a torch learner useful behind `$base_learner()`.
#' `as_learner_torch()` instead returns a first-class [`LearnerTorch`], i.e. one with a `$network`
#' field, a `$dataset()` method, the torch parameters (`epochs`, `batch_size`, ...) at the top level,
#' as well as marshaling, validation and internal tuning.
#'
#' @details
#' The neural network can only be built once the [`Task`][mlr3::Task] is known, because the input
#' shapes are derived from it. The returned learner therefore stores the `Graph` and runs it during
#' `$train()` to obtain the [`ModelDescriptor`] from which the network is created.
#'
#' The operators that configure the *training* of the model instead of the network are consumed
#' during the conversion, i.e. they are taken out of the graph and their configuration is moved to
#' the learner:
#'
#' * [`po("torch_loss")`][mlr_pipeops_torch_loss] becomes the learner's `$loss` (parameters `loss.*`).
#' * [`po("torch_optimizer")`][mlr_pipeops_torch_optimizer] becomes the learner's `$optimizer`
#'   (parameters `opt.*`).
#' * [`po("torch_callbacks")`][mlr_pipeops_torch_callbacks] becomes the learner's `$callbacks`
#'   (parameters `cb.<id>.*`).
#' * [`po("torch_model_classif")`][mlr_pipeops_torch_model_classif] and friends determine the
#'   `task_type` and their parameter values (`epochs`, `batch_size`, ...) become the learner's.
#'
#' The `loss`, `optimizer` and `callbacks` arguments take precedence over the corresponding operators
#' in the graph.
#' All remaining parameters of the graph are exposed in the learner's `$param_set` under the same
#' (`PipeOp`-id prefixed) names that the graph uses, so e.g. `nn_linear.out_features` keeps working.
#'
#' Operators that transform the [`Task`][mlr3::Task] before it reaches an ingress -- such as
#' [`po("scale")`][mlr3pipelines::mlr_pipeops_scale] or the
#' [preprocessing operators][PipeOpTaskPreprocTorch] -- cannot be part of the learner: they fit a
#' state during training and behave differently during prediction, which is what a
#' [`GraphLearner`][mlr3pipelines::GraphLearner] is for. Put them in front of the learner instead,
#' e.g. `po("scale") %>>% as_learner_torch(graph)`.
#' Only operators that route features without learning anything from them -- such as
#' [`po("select")`][mlr3pipelines::mlr_pipeops_select], which is how a network with several ingress
#' operators splits the task -- may precede an ingress.
#'
#' @param x (any)\cr
#'   The object to convert, e.g. a [`Graph`][mlr3pipelines::Graph].
#' @param ... (any)\cr
#'   Parameter values passed to the created learner, e.g. `epochs = 10`.
#' @param task_type (`character(1)`)\cr
#'   The task type of the learner.
#'   Can be omitted if the graph contains a [`PipeOpTorchModel`], which then determines it.
#' @param id (`character(1)`)\cr
#'   The id of the learner. Defaults to `"<task_type>.graph"`.
#' @template param_optimizer
#' @template param_loss
#' @template param_callbacks
#'
#' @return [`LearnerTorch`]
#' @family Graph Network
#' @family Learner
#' @include LearnerTorch.R
#' @export
#' @examplesIf torch::torch_is_installed()
#' graph = po("torch_ingress_num") %>>%
#'   nn("linear", out_features = 10) %>>%
#'   nn("relu") %>>%
#'   nn("head") %>>%
#'   po("torch_loss", "cross_entropy") %>>%
#'   po("torch_optimizer", "adam", lr = 0.1)
#'
#' learner = as_learner_torch(graph, task_type = "classif", epochs = 1, batch_size = 32)
#' learner$id
#' # the parameters of the graph are available at the top level
#' learner$param_set$set_values(linear.out_features = 20, opt.lr = 0.01)
#'
#' task = tsk("iris")
#' learner$train(task)
#' learner$network
#' learner$predict(task)
as_learner_torch = function(x, ...) {
  UseMethod("as_learner_torch")
}

#' @rdname as_learner_torch
#' @export
as_learner_torch.Graph = function(x, task_type = NULL, id = NULL, optimizer = NULL, loss = NULL, # nolint
  callbacks = NULL, ...) {
  LearnerTorchGraph$new(
    graph = x,
    task_type = task_type,
    id = id,
    optimizer = optimizer,
    loss = loss,
    callbacks = callbacks,
    param_vals = list(...)
  )
}

#' @rdname as_learner_torch
#' @export
as_learner_torch.PipeOp = function(x, ...) { # nolint
  as_learner_torch(as_graph(x), ...)
}

#' @rdname as_learner_torch
#' @export
as_learner_torch.GraphLearner = function(x, task_type = x$task_type, id = NULL, ...) { # nolint
  as_learner_torch(x$graph, task_type = task_type, id = id, ...)
}

# The learner returned by `as_learner_torch()`. It is intentionally not exported and not part of the
# `mlr_learners` dictionary, because it cannot be constructed without a `Graph`.
LearnerTorchGraph = R6Class("LearnerTorchGraph",
  inherit = LearnerTorch,
  public = list(
    initialize = function(graph, task_type = NULL, id = NULL, optimizer = NULL, loss = NULL,
      callbacks = NULL, param_vals = list()) {
      graph = as_graph(assert_r6(graph, "Graph"), clone = TRUE)
      assert_string(id, null.ok = TRUE)
      assert_list(param_vals, names = "unique")

      ingress_ids = graph_ingress_ids(graph)
      if (!length(ingress_ids)) {
        stopf("Graph cannot be converted to a LearnerTorch because it contains no PipeOpTorchIngress, add e.g. po(\"torch_ingress_num\") to its start.") # nolint
      }
      if (nrow(graph$output) != 1L) {
        stopf("Graph cannot be converted to a LearnerTorch because it has %i output channels, but exactly one is required.", nrow(graph$output)) # nolint
      }
      pre_ingress = setdiff(graph_ingress_ancestors(graph, ingress_ids), ingress_ids)
      unsupported = pre_ingress[map_lgl(pre_ingress, function(po_id) {
        !any(class(graph$pipeops[[po_id]]) %in% mlr3torch_routing_pipeops)
      })]
      if (length(unsupported)) {
        stopf("PipeOp(s) %s cannot be part of a LearnerTorch: an operator before a PipeOpTorchIngress is applied to the prediction task as well, so it must not fit anything on the training data. Put it in front of the learner instead, e.g. po(\"scale\") %%>>%% as_learner_torch(graph).", paste0("'", unsupported, "'", collapse = ", ")) # nolint
      }

      po_model = graph_single_pipeop(graph, "PipeOpTorchModel")
      task_type_graph = if (!is.null(po_model)) get_private(po_model)$.task_type
      if (is.null(task_type)) {
        if (is.null(task_type_graph)) {
          stopf("Cannot infer the task type of the graph, pass it via the `task_type` argument.")
        }
        task_type = task_type_graph
      } else {
        assert_choice(task_type, mlr_reflections$task_types$type)
        if (!is.null(task_type_graph) && task_type != task_type_graph) {
          stopf("Task type '%s' was requested, but the graph contains PipeOp '%s' with task type '%s'.", task_type, po_model$id, task_type_graph) # nolint
        }
      }

      # the configuration of the network's *training* moves from the graph to the learner, where it
      # is exposed under the standard LearnerTorch names ('epochs', 'loss.*', 'opt.*', 'cb.*');
      # leaving the operators in the graph would expose the same settings twice under two names
      po_loss = graph_single_pipeop(graph, "PipeOpTorchLoss")
      po_optimizer = graph_single_pipeop(graph, "PipeOpTorchOptimizer")
      pos_callbacks = graph_pipeops(graph, "PipeOpTorchCallbacks")
      loss = loss %??% (if (!is.null(po_loss)) get_private(po_loss)$.loss)
      optimizer = optimizer %??% (if (!is.null(po_optimizer)) get_private(po_optimizer)$.optimizer)
      callbacks = callbacks %??%
        unlist(map(pos_callbacks, function(po) get_private(po)$.callbacks), recursive = FALSE)
      model_param_vals = if (!is.null(po_model)) po_model$param_set$values else list()

      consumed = c(names(pos_callbacks), map_chr(discard(list(po_loss, po_optimizer, po_model), is.null), "id"))
      # `PipeOpNOP` passes the ModelDescriptor through unchanged and has no parameters, so replacing
      # the consumed operators keeps the graph structure (and hence all edges) intact
      for (po_id in consumed) {
        graph$pipeops[[po_id]] = po("nop", id = po_id)
      }
      if (length(consumed)) {
        graph$.__enclos_env__$private$.param_set = NULL
      }
      if (graph$output$train %nin% c("ModelDescriptor", "*")) {
        stopf("Graph cannot be converted to a LearnerTorch because its output is of type '%s', but a ModelDescriptor is required.", graph$output$train) # nolint
      }

      private$.graph = graph
      private$.ingress_ids = ingress_ids

      super$initialize(
        id = id %??% paste0(task_type, ".graph"),
        task_type = task_type,
        param_set = alist(private$.graph$param_set),
        loss = loss,
        optimizer = optimizer,
        callbacks = callbacks %??% list(),
        packages = graph$packages,
        feature_types = graph_feature_types(graph, ingress_ids),
        label = "Graph Network",
        man = "mlr3torch::as_learner_torch",
        jittable = TRUE
      )

      self$param_set$set_values(.values = insert_named(model_param_vals, param_vals))
    }
  ),
  active = list(
    # The graph that defines the network. Its parameters are exposed in `$param_set`.
    graph = function(rhs) {
      assert_ro_binding(rhs)
      private$.graph
    }
  ),
  private = list(
    .graph = NULL,
    .ingress_ids = NULL,
    .network = function(task, param_vals) {
      md = private$.model_descriptor(task)
      model_descriptor_to_module(md, output_pointers = list(md$pointer), list_output = FALSE)
    },
    .ingress_tokens = function(task, param_vals) {
      graph = graph_ingress_part(private$.graph, private$.ingress_ids)
      mds = keep(graph$train(task), function(x) test_class(x, "ModelDescriptor"))
      tokens = do.call(c, c(list(list()), unname(map(mds, "ingress"))))
      assert_names(names(tokens), type = "unique")
      tokens
    },
    .model_descriptor = function(task) {
      # the graph is stateful (it stores the built modules) and training it here must not change the
      # learner, which is why we run a copy
      md = private$.graph$clone(deep = TRUE)$train(task)[[1L]]
      if (!test_class(md, "ModelDescriptor")) {
        stopf("Learner '%s': the graph produced an object of class '%s' instead of a ModelDescriptor.", self$id, class(md)[[1L]]) # nolint
      }
      md
    },
    .additional_phash_input = function() {
      list(private$.graph$phash, self$task_type, self$feature_types, self$properties, self$packages)
    }
  )
)

# Operators that only route features around instead of learning something from them. They are the
# only ones a `LearnerTorch` can run before its ingress, see the check in `LearnerTorchGraph`.
mlr3torch_routing_pipeops = c("PipeOpSelect", "PipeOpNOP", "PipeOpBranch", "PipeOpUnbranch",
  "PipeOpCopy", "PipeOpFeatureUnion")

graph_pipeops = function(graph, class) {
  keep(graph$pipeops, function(po) test_class(po, class))
}

graph_ingress_ids = function(graph) {
  names(graph_pipeops(graph, "PipeOpTorchIngress"))
}

graph_single_pipeop = function(graph, class) {
  pos = graph_pipeops(graph, class)
  if (length(pos) > 1L) {
    stopf("Graph cannot be converted to a LearnerTorch because it contains more than one %s: %s.", class, paste0("'", names(pos), "'", collapse = ", ")) # nolint
  }
  if (length(pos)) pos[[1L]]
}

# Only the ingress operators can restrict which feature types the learner accepts, and only when
# nothing preprocesses the task before them -- otherwise the features the ingress sees are not the
# features of the task. Every ingress sees the whole task, hence the intersection.
graph_feature_types = function(graph, ingress_ids) {
  ingress_pos = graph$pipeops[ingress_ids]
  is_source = !any(graph$edges$dst_id %in% ingress_ids)
  if (!is_source) {
    return(unname(mlr_reflections$task_feature_types))
  }
  Reduce(intersect, map(ingress_pos, "feature_types"))
}

graph_ingress_ancestors = function(graph, ingress_ids) {
  edges = graph$edges
  ids = ingress_ids
  repeat {
    parents = if (nrow(edges)) unique(edges$src_id[edges$dst_id %in% ids]) else character(0)
    new_ids = setdiff(parents, ids)
    if (!length(new_ids)) break
    ids = c(ids, new_ids)
  }
  ids
}

# The ingress tokens are needed to build the dataloader, also during prediction, where the network
# already exists. Because only `PipeOpTorchIngress` creates them, it suffices to run the part of the
# graph that ends in the ingress operators, which avoids building the network a second time.
graph_ingress_part = function(graph, ingress_ids) {
  edges = graph$edges
  ids = graph_ingress_ancestors(graph, ingress_ids)
  part = Graph$new()
  for (po_id in ids) {
    part$add_pipeop(graph$pipeops[[po_id]], clone = TRUE)
  }
  sub_edges = edges[edges$src_id %in% ids & edges$dst_id %in% ids, ]
  for (i in seq_len(nrow(sub_edges))) {
    part$add_edge(
      src_id = sub_edges$src_id[[i]], src_channel = sub_edges$src_channel[[i]],
      dst_id = sub_edges$dst_id[[i]], dst_channel = sub_edges$dst_channel[[i]]
    )
  }
  part
}
