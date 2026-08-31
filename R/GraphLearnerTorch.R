#' @title Convert a Graph to a Torch Learner
#'
#' @description
#' Converts a [`Graph`][mlr3pipelines::Graph] representing a deep learning pipeline into a
#' [`GraphLearnerTorch`].
#' The advantage over using [`as_learner()`] is that the resulting learner exposes methods like
#' `$dataset()` and fields like `$network`.
#'
#' @param x (any)\cr
#'   The object to convert, e.g. a [`Graph`][mlr3pipelines::Graph].
#' @param id (`character(1)`)\cr
#'   The id of the learner. Defaults to `"<task_type>.graph"`.
#' @param ... (any)\cr
#'   Unused.
#'
#' @return [`GraphLearner`][mlr3pipelines::GraphLearner]
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
#'   po("torch_optimizer", "adam", lr = 0.1) %>>%
#'   po("torch_model_classif", epochs = 1, batch_size = 32)
#'
#' learner = as_learner_torch(graph)
#' learner$id
#' learner$param_set$set_values(linear.out_features = 20, torch_optimizer.lr = 0.01)
#'
#' task = tsk("iris")
#' learner$train(task)
#' learner$network
#' learner$predict(task)
#'
#' learner$dataset(task, "train")
as_learner_torch = function(x, ...) {
  UseMethod("as_learner_torch")
}

#' @rdname as_learner_torch
#' @export
as_learner_torch.Graph = function(x, id = NULL, ...) { # nolint
  po_model = graph_single_pipeop(x, "PipeOpTorchModel")
  if (is.null(po_model)) {
    stopf("Graph is not a torch learner graph because it contains no PipeOpTorchModel, add e.g. po(\"torch_model_classif\") to its end.") # nolint
  }
  task_type = get_private(po_model)$.task_type

  GraphLearnerTorch$new(
    graph = x,
    id = id %??% paste0(task_type, ".graph"),
    task_type = task_type
  )
}

#' @rdname as_learner_torch
#' @export
as_learner_torch.PipeOp = function(x, ...) { # nolint
  as_learner_torch(as_graph(x), ...)
}

#' @rdname as_learner_torch
#' @export
as_learner_torch.GraphLearner = function(x, id = x$id, ...) { # nolint
  as_learner_torch(x$graph, id = id, ...)
}

#' @title Graph Learner for Torch Networks
#'
#' @description
#' The [`GraphLearner`][mlr3pipelines::GraphLearner] that [`as_learner_torch()`] returns.
#' On top of a `GraphLearner` it has the `$network`, `$loss`, `$optimizer` and `$callbacks` fields
#' and the `$dataset()` method of a [`LearnerTorch`].
#' It is not in the [`mlr_learners`][mlr3::mlr_learners] dictionary, because it cannot be
#' constructed without a graph.
#'
#' @family Learner
#' @export
GraphLearnerTorch = R6Class("GraphLearnerTorch",
  inherit = GraphLearner,
  public = list(
    #' @description
    #' Create the dataset for a task, i.e. the tensors that are fed to the network.
    #' @param task [`Task`][mlr3::Task]\cr
    #'   The task.
    #' @param stage (`character(1)`)\cr
    #'   Whether to create the dataset the way `$train()` does (`"train"`) or the way `$predict()`
    #'   does (`"predict"`).
    #'   Defaults to `"predict"` for a trained learner and to `"train"` otherwise, because the
    #'   prediction phase reuses the state that the operators before the ingress fitted during
    #'   training.
    #' @return [`dataset`][torch::dataset]
    dataset = function(task, stage = if (is.null(self$model)) "train" else "predict") {
      assert_task(task)
      assert_choice(stage, c("train", "predict"))
      if (stage == "train") {
        # the operators behind the ingress only build the network's modules, which the data does not
        # need, so the graph is cut there instead of in front of the `PipeOpTorchModel`
        mds = graph_ingress_part(self$graph)$train(task)
        md = Reduce(model_descriptor_union, mds)
        task_dataset(
          md$task,
          feature_ingress_tokens = md$ingress,
          target_batchgetter = get_target_batchgetter(md$task)
        )
      } else {
        if (is.null(self$model)) {
          stopf("Learner '%s' must be trained before the data of the prediction phase can be created, because the operators before its ingress have not been fitted yet.", self$id) # nolint
        }
        # the ingress operators pass the task through during prediction, so these are the tasks that
        # the operators in front of them produced; they are merged just like `model_descriptor_union()`
        # merges the tasks of several ingress paths during training
        tasks = graph_ingress_part(self$graph_model)$predict(task)
        self$base_learner()$dataset(Reduce(function(task1, task2) {
          if (identical(task1, task2)) task1 else PipeOpFeatureUnion$new()$train(list(task1, task2))[[1L]] # nolint
        }, tasks))
      }
    }
  ),
  active = list(
    #' @field network ([`nn_module`][torch::nn_module])\cr
    #' The network of the trained learner, i.e. `$base_learner()$network`.
    network = function(rhs) {
      assert_ro_binding(rhs)
      self$base_learner()$network
    },
    #' @field loss ([`TorchLoss`])\cr
    #' The torch loss, i.e. the one of the graph's [`PipeOpTorchLoss`][mlr_pipeops_torch_loss].
    #' Read-only, the graph configures it.
    loss = function(rhs) {
      assert_ro_binding(rhs)
      private$.configuration("PipeOpTorchLoss", ".loss")
    },
    #' @field optimizer ([`TorchOptimizer`])\cr
    #' The torch optimizer, i.e. the one of the graph's
    #' [`PipeOpTorchOptimizer`][mlr_pipeops_torch_optimizer].
    #' Read-only, the graph configures it.
    optimizer = function(rhs) {
      assert_ro_binding(rhs)
      private$.configuration("PipeOpTorchOptimizer", ".optimizer")
    },
    #' @field callbacks (`list()` of [`TorchCallback`]s)\cr
    #' The callbacks, i.e. those of the graph's
    #' [`PipeOpTorchCallbacks`][mlr_pipeops_torch_callbacks].
    #' Read-only, the graph configures them.
    callbacks = function(rhs) {
      assert_ro_binding(rhs)
      private$.configuration("PipeOpTorchCallbacks", ".callbacks") %??% list()
    }
  ),
  private = list(
    # The object that the operator of the given class configures, e.g. the `TorchLoss` of the
    # `PipeOpTorchLoss`, or `NULL` if the graph contains no such operator.
    .configuration = function(class, field) {
      po_config = graph_single_pipeop(private$.graph, class)
      if (!is.null(po_config)) get_private(po_config)[[field]]
    }
  )
)

# The unique `PipeOp` of the given class, or `NULL` if the graph contains none.
graph_single_pipeop = function(graph, class) {
  pos = keep(graph$pipeops, function(po) test_class(po, class))
  if (length(pos) > 1L) {
    stopf("Graph contains more than one %s: %s.", class, paste0("'", names(pos), "'", collapse = ", "))
  }
  if (length(pos)) pos[[1L]]
}

# The part of the graph that turns the task into the data of the network, i.e. the
# `PipeOpTorchIngress` operators and everything that feeds into them.
graph_ingress_part = function(graph) {
  ids = names(keep(graph$pipeops, function(po) test_class(po, "PipeOpTorchIngress")))
  repeat {
    parents = unique(graph$edges[get("dst_id") %in% ids][["src_id"]])
    new_ids = setdiff(parents, ids)
    if (!length(new_ids)) break
    ids = c(ids, new_ids)
  }

  part = Graph$new()
  for (id in ids) {
    part$add_pipeop(graph$pipeops[[id]], clone = TRUE)
  }
  edges = graph$edges[get("src_id") %in% ids & get("dst_id") %in% ids]
  for (i in seq_len(nrow(edges))) {
    part$add_edge(
      src_id = edges$src_id[[i]], src_channel = edges$src_channel[[i]],
      dst_id = edges$dst_id[[i]], dst_channel = edges$dst_channel[[i]]
    )
  }
  part
}
