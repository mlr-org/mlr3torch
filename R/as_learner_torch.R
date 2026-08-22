#' @title Convert a Graph to a Torch Learner
#'
#' @description
#' Converts a [`Graph`][mlr3pipelines::Graph] that is built from [`PipeOpTorch`] operators into a
#' [`GraphLearner`][mlr3pipelines::GraphLearner] that trains the network it defines.
#'
#' In contrast to [`as_learner()`][mlr3::as_learner], the terminal [`PipeOpTorchModel`] is appended
#' to the graph if it does not contain one already, and the returned learner has the `$network`,
#' `$loss`, `$optimizer` and `$callbacks` fields and the `$dataset()` method of a torch learner.
#'
#' @details
#' The learner is a [`GraphLearner`][mlr3pipelines::GraphLearner], i.e. the parameters of the graph
#' are exposed in `$param_set` under their (`PipeOp`-id prefixed) graph names, so the number of
#' epochs is e.g. `torch_model_classif.epochs` and not `epochs`.
#' Because the parameters of the torch model are the ones that are usually configured, `...` sets
#' the parameters of the [`PipeOpTorchModel`], i.e. `as_learner_torch(graph, epochs = 10)` is the
#' same as `graph %>>% po("torch_model_classif", epochs = 10)` converted with `as_learner()`.
#'
#' The loss, the optimizer and the callbacks are configured in the graph via
#' [`po("torch_loss")`][mlr_pipeops_torch_loss], [`po("torch_optimizer")`][mlr_pipeops_torch_optimizer]
#' and [`po("torch_callbacks")`][mlr_pipeops_torch_callbacks], of which the graph may contain at most
#' one each.
#' Just like for a [`LearnerTorch`], they are available via the `$loss`, `$optimizer` and
#' `$callbacks` fields, which also replace the corresponding operator when assigned to.
#' The arguments of the same name configure them during construction and take precedence over what
#' the graph configures.
#' When neither does, the defaults of [`LearnerTorch`] are used, i.e. the cross-entropy
#' (classification) or mean-squared-error (regression) loss and the Adam optimizer.
#'
#' The neural network is only built during `$train()`, because the input shapes are derived from the
#' [`Task`][mlr3::Task].
#' Afterwards it is available via `$network`, which is a shortcut for `$base_learner()$network`.
#' `$dataset(task)` returns the tensors of the prediction phase for a trained learner, pass
#' `train = TRUE` for those of the training phase.
#' The two differ when the graph transforms the task before it reaches an ingress -- such as
#' [`po("scale")`][mlr3pipelines::mlr_pipeops_scale] or the
#' [preprocessing operators][PipeOpTaskPreprocTorch] -- because the prediction phase reuses the
#' state that these operators fitted during training.
#'
#' @param x (any)\cr
#'   The object to convert, e.g. a [`Graph`][mlr3pipelines::Graph].
#' @param ... (any)\cr
#'   Parameter values for the [`PipeOpTorchModel`], e.g. `epochs = 10`.
#' @param task_type (`character(1)`)\cr
#'   The task type of the learner.
#'   Can be omitted if the graph already contains a [`PipeOpTorchModel`], which then determines it.
#' @param id (`character(1)`)\cr
#'   The id of the learner. Defaults to `"<task_type>.graph"`.
#' @template param_optimizer
#' @template param_loss
#' @template param_callbacks
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
#'   po("torch_optimizer", "adam", lr = 0.1)
#'
#' learner = as_learner_torch(graph, task_type = "classif", epochs = 1, batch_size = 32)
#' learner$id
#' learner$param_set$set_values(linear.out_features = 20, torch_optimizer.lr = 0.01)
#'
#' task = tsk("iris")
#' learner$train(task)
#' learner$network
#' learner$predict(task)
#'
#' # po("scale") is part of the learner: it was fitted during $train() and the prediction data
#' # is standardized with those statistics
#' learner$dataset(task)
as_learner_torch = function(x, ...) {
  UseMethod("as_learner_torch")
}

#' @rdname as_learner_torch
#' @export
as_learner_torch.Graph = function(x, task_type = NULL, id = NULL, optimizer = NULL, loss = NULL, # nolint
  callbacks = NULL, ...) {
  GraphLearnerTorch$new(
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
as_learner_torch.GraphLearner = function(x, task_type = x$task_type, id = x$id, ...) { # nolint
  as_learner_torch(x$graph, task_type = task_type, id = id, ...)
}

# The learner returned by `as_learner_torch()`. It is a `GraphLearner` that knows where the network
# and its data live, hence it is neither exported nor part of the `mlr_learners` dictionary: it
# cannot be constructed without a `Graph`.
GraphLearnerTorch = R6Class("GraphLearnerTorch",
  inherit = GraphLearner,
  public = list(
    initialize = function(graph, task_type = NULL, id = NULL, optimizer = NULL, loss = NULL,
      callbacks = NULL, param_vals = list()) {
      graph = as_graph(assert_r6(graph, "Graph"), clone = TRUE)
      assert_list(param_vals, names = "unique")

      if (!some(graph$pipeops, function(po) test_class(po, "PipeOpTorchIngress"))) {
        stopf("Graph cannot be converted to a torch learner because it contains no PipeOpTorchIngress, add e.g. po(\"torch_ingress_num\") to its start.") # nolint
      }

      po_model = graph_torch_model(graph)
      if (is.null(po_model)) {
        if (is.null(task_type)) {
          stopf("Cannot infer the task type of the graph, pass it via the `task_type` argument.")
        }
        assert_choice(task_type, mlr_reflections$task_types$type)
        po_model = PipeOpTorchModel$new(task_type = task_type, id = paste0("torch_model_", task_type))
      } else {
        task_type_graph = get_private(po_model)$.task_type
        if (is.null(task_type)) {
          task_type = task_type_graph
        } else if (task_type != task_type_graph) {
          stopf("Task type '%s' was requested, but the graph contains PipeOp '%s' with task type '%s'.", task_type, po_model$id, task_type_graph) # nolint
        }
        # the model operator is re-appended below, so that the operators that configure the training
        # end up in front of it
        graph = graph_before_model(graph)
      }
      po_model$param_set$set_values(.values = param_vals)

      # the loss and the optimizer are part of the ModelDescriptor, so `PipeOpTorchModel` cannot
      # fall back to the defaults of `LearnerTorch` itself and the graph has to configure them
      graph = graph_default(graph, "PipeOpTorchLoss", "torch_loss",
        switch(task_type, classif = "cross_entropy", regr = "mse", NULL))
      graph = graph_default(graph, "PipeOpTorchOptimizer", "torch_optimizer", "adam")
      graph = graph %>>% po_model

      super$initialize(
        graph = graph,
        id = id %??% paste0(task_type, ".graph"),
        task_type = task_type,
        clone_graph = FALSE
      )

      # the arguments take precedence over what the graph configures
      if (!is.null(loss)) self$loss = loss
      if (!is.null(optimizer)) self$optimizer = optimizer
      if (!is.null(callbacks)) self$callbacks = callbacks
    },
    #' @description
    #' Create the dataset for a task, i.e. the tensors that are fed to the network.
    #' @param task [`Task`][mlr3::Task]\cr
    #'   The task.
    #' @param train (`logical(1)`)\cr
    #'   Whether to create the dataset the way `$train()` does (`TRUE`) or the way `$predict()` does
    #'   (`FALSE`).
    #'   Defaults to `FALSE` for a trained learner and to `TRUE` otherwise, because the prediction
    #'   phase reuses the state that the operators before the ingress fitted during training.
    #' @return [`dataset`][torch::dataset]
    dataset = function(task, train = is.null(self$model)) {
      assert_task(task)
      assert_flag(train)
      if (train) {
        md = graph_before_model(self$graph)$train(task)[[1L]]
        task_dataset(
          md$task,
          feature_ingress_tokens = md$ingress,
          target_batchgetter = get_target_batchgetter(md$task)
        )
      } else {
        if (is.null(self$model)) {
          stopf("Learner '%s' must be trained before the data of the prediction phase can be created, because the operators before its ingress have not been fitted yet.", self$id) # nolint
        }
        # in the prediction phase the operators pass the task through, so this is the task that the
        # PipeOpTorchModel -- and hence its LearnerTorchModel -- receives
        self$base_learner()$dataset(graph_before_model(self$graph_model)$predict(task)[[1L]])
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
    loss = function(rhs) {
      if (!missing(rhs)) {
        rhs = as_torch_loss(rhs, clone = TRUE)
        assert_choice(self$task_type, rhs$task_types)
        private$.configure("PipeOpTorchLoss", "torch_loss", rhs)
      }
      private$.configuration("PipeOpTorchLoss", ".loss")
    },
    #' @field optimizer ([`TorchOptimizer`])\cr
    #' The torch optimizer, i.e. the one of the graph's
    #' [`PipeOpTorchOptimizer`][mlr_pipeops_torch_optimizer].
    optimizer = function(rhs) {
      if (!missing(rhs)) {
        private$.configure("PipeOpTorchOptimizer", "torch_optimizer", as_torch_optimizer(rhs, clone = TRUE))
      }
      private$.configuration("PipeOpTorchOptimizer", ".optimizer")
    },
    #' @field callbacks (`list()` of [`TorchCallback`]s)\cr
    #' The callbacks, i.e. those of the graph's
    #' [`PipeOpTorchCallbacks`][mlr_pipeops_torch_callbacks].
    callbacks = function(rhs) {
      if (!missing(rhs)) {
        private$.configure("PipeOpTorchCallbacks", "torch_callbacks", as_torch_callbacks(rhs, clone = TRUE))
      }
      private$.configuration("PipeOpTorchCallbacks", ".callbacks") %??% list()
    }
  ),
  private = list(
    # The object that the operator of the given class configures, e.g. the `TorchLoss` of the
    # `PipeOpTorchLoss`, or `NULL` if the graph contains no such operator.
    .configuration = function(class, field) {
      po_config = graph_single_pipeop(private$.graph, class)
      if (!is.null(po_config)) get_private(po_config)[[field]]
    },
    # Configure that aspect of the training. The whole operator is replaced instead of the object it
    # holds, because the operator's ParamSet is the one of that object -- which is also why the
    # parameter values of the operator are lost, just like they are when the loss of a `LearnerTorch`
    # is replaced.
    .configure = function(class, key, value) {
      graph = private$.graph
      po_config = graph_single_pipeop(graph, class)
      if (is.null(po_config)) {
        # the operator has to come in front of the model, which is the last operator of the graph
        private$.graph = graph_before_model(graph) %>>% po(key, value) %>>% graph_torch_model(graph)
      } else {
        graph$pipeops[[po_config$id]] = po(key, value, id = po_config$id)
        # the graph caches the collection of the parameter sets of its operators
        graph$.__enclos_env__$private$.param_set = NULL
      }
      self$packages = union(self$packages, private$.graph$packages)
      invisible(NULL)
    }
  )
)

# Append the operator that configures one aspect of the training with the default value, unless the
# graph configures it already.
graph_default = function(graph, class, key, default) {
  if (is.null(default) || !is.null(graph_single_pipeop(graph, class))) graph else graph %>>% po(key, default)
}

# The unique `PipeOp` of the given class, or `NULL` if the graph contains none.
graph_single_pipeop = function(graph, class) {
  pos = keep(graph$pipeops, function(po) test_class(po, class))
  if (length(pos) > 1L) {
    stopf("Graph cannot be converted to a torch learner because it contains more than one %s: %s.", class, paste0("'", names(pos), "'", collapse = ", ")) # nolint
  }
  if (length(pos)) pos[[1L]]
}

graph_torch_model = function(graph) {
  graph_single_pipeop(graph, "PipeOpTorchModel")
}

# Everything but the terminal `PipeOpTorchModel`, i.e. the part of the graph that turns the task
# into the `ModelDescriptor` (during training) or into the task that the network's data is built
# from (during prediction).
graph_before_model = function(graph) {
  id = graph_torch_model(graph)$id
  graph = graph$clone(deep = TRUE)
  graph$pipeops[[id]] = NULL
  graph$edges = graph$edges[get("src_id") != id & get("dst_id") != id]
  graph
}
