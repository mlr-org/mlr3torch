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
#' [`po("scale")`][mlr3pipelines::mlr_pipeops_scale], [`po("select")`][mlr3pipelines::mlr_pipeops_select]
#' (which is how a network with several ingress operators splits the task) or the
#' [preprocessing operators][PipeOpTaskPreprocTorch] -- are part of the learner and keep their
#' [`PipeOp`][mlr3pipelines::PipeOp] semantics: they are trained during `$train()` and the state they
#' fitted is reused during `$predict()`, so e.g. `po("scale")` standardizes the prediction data with
#' the training statistics and an augmentation with `stages = "train"` is not applied during
#' prediction.
#' The fitted states are stored in `$model$ingress$states`.
#' `$dataset(task)` returns the tensors of the prediction phase for a trained learner, pass
#' `train = TRUE` for those of the training phase.
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
#' graph = po("scale") %>>%
#'   po("torch_ingress_num") %>>%
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
#'
#' # po("scale") is part of the learner: it was fitted during $train() and the prediction data
#' # is standardized with those statistics
#' learner$model$ingress$states$scale$center
#' learner$dataset(task)
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
      private$.part_ids = graph_ingress_ancestors(graph, ingress_ids)

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
    # ids of the operators up to and including the ingress, i.e. the part that turns the task into
    # the one the network is trained on
    .part_ids = NULL,
    # the ingress tokens of the task that `.prepare_task()` last returned; always a duplicate of
    # what the model or the descriptor below holds, so keeping it around loses nothing
    .ingress_tokens_ = NULL,
    # hands the built network from `.train()`, which runs the graph, to `.network()`, which is
    # called further down in `super$.train()`; it holds torch modules, hence the `on.exit()` there
    .md = NULL,
    .train = function(task) {
      # one run of the graph does everything the training phase needs: it fits the states of the
      # operators before the ingress, creates the ingress tokens, and builds the modules
      graph = private$.graph$clone(deep = TRUE)
      md = graph$train(task)[[1L]]
      if (!test_class(md, "ModelDescriptor")) {
        stopf("Learner '%s': the graph produced an object of class '%s' instead of a ModelDescriptor.", self$id, class(md)[[1L]]) # nolint
      }
      private$.md = md
      private$.ingress_tokens_ = md$ingress
      on.exit({private$.md = NULL}, add = TRUE)

      # `md$task` is the task after the operators before the ingress ran, i.e. the one the network
      # is built for; its `internal_valid_task` was transformed in predict mode by those operators
      model = super$.train(md$task)
      model$ingress = list(
        tokens = md$ingress,
        states = map(graph$pipeops[private$.part_ids], "state")
      )
      model
    },
    .predict = function(task) {
      super$.predict(private$.prepare_task(task, train = FALSE))
    },
    .prepare_task = function(task, train) {
      part = graph_ingress_part(private$.graph, private$.ingress_ids)
      if (train) {
        mds = keep(part$train(task), function(x) test_class(x, "ModelDescriptor"))
        md = Reduce(model_descriptor_union, mds)
        private$.ingress_tokens_ = md$ingress
        return(md$task)
      }
      states = self$model$ingress$states
      if (is.null(states)) {
        stopf("Learner '%s' must be trained before the prediction phase's data can be created, because the operators before its ingress have not been fitted yet.", self$id) # nolint
      }
      for (po_id in names(states)) {
        part$pipeops[[po_id]]$state = states[[po_id]]
      }
      # the ingress operators pass the task through during prediction, so this is the task the
      # operators before them produced; it is merged the way `model_descriptor_union()` merges the
      # tasks of several ingress paths during training
      tasks = keep(part$predict(task), function(x) test_class(x, "Task"))
      prepared = Reduce(function(t1, t2) {
        if (identical(t1, t2)) t1 else PipeOpFeatureUnion$new()$train(list(t1, t2))[[1L]]
      }, tasks)
      if (!identical(prepared$row_ids, task$row_ids)) {
        stopf("Learner '%s': the operators before its ingress changed the rows of task '%s' during prediction.", self$id, task$id) # nolint
      }
      private$.ingress_tokens_ = self$model$ingress$tokens
      prepared
    },
    .network = function(task, param_vals) {
      if (is.null(private$.md)) {
        stopf("Learner '%s' can only build its network during training.", self$id)
      }
      network = model_descriptor_to_module(private$.md, output_pointers = list(private$.md$pointer),
        list_output = FALSE)
      # the graph built the modules before `$.train()` seeded torch's generator, so their weights
      # were drawn outside the seeded region; re-initializing here puts them back under the seed,
      # exactly like `PipeOpTorchModel` does for the GraphLearner route
      network$reset_parameters()
      network
    },
    .ingress_tokens = function(task, param_vals) {
      tokens = private$.ingress_tokens_ %??% self$model$ingress$tokens
      if (is.null(tokens)) {
        stopf("Learner '%s' has no ingress tokens, use $dataset() only on a trained learner or with train = TRUE.", self$id) # nolint
      }
      tokens
    },
    .additional_phash_input = function() {
      list(private$.graph$phash, self$task_type, self$feature_types, self$properties, self$packages)
    }
  )
)

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

# The part of the graph up to and including the ingress operators, i.e. everything that turns the
# task into the task the network is trained on. It is what the learner has to run again during
# prediction; the operators behind the ingress only concern the network, which by then exists.
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
