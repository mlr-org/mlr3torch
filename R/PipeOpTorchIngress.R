#' @title Entrypoint to Torch Network
#'
#' @name mlr_pipeops_torch_ingress
#'
#' @description
#' Use this as entry-point to mlr3torch-networks.
#' Unless you are an advanced user, you should not need to use this directly but [`PipeOpTorchIngressNumeric`],
#' [`PipeOpTorchIngressCategorical`] or [`PipeOpTorchIngressLazyTensor`].
#'
#' @template pipeop_torch_channels_default
#' @section State:
#' The state is set to the input shape.
#'
#' @section Parameters:
#' Defined by the construction argument `param_set`.
#!'
#' @section Internals:
#' Creates an object of class [`TorchIngressToken`] for the given task.
#' The purpuse of this is to store the information on how to construct the torch dataloader from the task for this
#' entry point of the network.
#'
#' @family PipeOps
#' @family Graph Network
#' @export
PipeOpTorchIngress = R6Class("PipeOpTorchIngress",
  inherit = PipeOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @template param_param_set
    #' @template param_packages
    #' @template param_feature_types
    initialize = function(id, param_set = ps(), param_vals = list(), packages = character(0), feature_types) {
      private$.feature_types = assert_subset(feature_types, mlr_reflections$task_feature_types)

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = data.table(name = "input", train = "Task", predict = "Task"),
        output = data.table(name = "output", train = "ModelDescriptor", predict = "Task"),
        packages = union(packages, "mlr3torch")
      )
     }
  ),
  private = list(
    .get_batchgetter = function(task, param_vals) stop("private$.get_batchgetter() must be implemented."),
    .shape = function(task, param_vals) stop("private$.shape() must be implemented."),
    .feature_types = NULL,
    .train = function(inputs) {
      pv = self$param_set$get_values()
      task = inputs[[1]]
      if (any(task$missings(task$feature_names))) {
        # NAs are converted to their underlying machine representation when calling `torch_tensor()`
        # https://github.com/mlverse/torch/issues/933
        stopf("No missing values allowed in task '%s'.", task$id)
      }
      param_vals = self$param_set$get_values()
      graph = as_graph(po("nop", id = self$id), clone = FALSE)
      batchgetter = private$.get_batchgetter(task, param_vals)

      # In case the user is tempted to do things that will break in bad ways...
      # But this test could also be left out.
      forbidden_env = parent.env(environment())
      test_env = environment(batchgetter)
      while (!isNamespace(test_env) && !identical(test_env, .GlobalEnv) && !identical(test_env, emptyenv())) {
        if (identical(test_env, forbidden_env)) {
          stopf(".get_batchgetter() must not return a member of the PipeOpTorchIngress object itself.")
        }
        test_env = parent.env(test_env)
      }

      if (!all(task$feature_types$type %in% self$feature_types)) {
        stopf("Task contains features of type %s, but only %s are allowed; Consider using po(\"select\").",
          paste0(unique(task$feature_types$type[!(task$feature_types$type %in% self$feature_types)]), collapse = ", "),
          paste0(self$feature_types, collapse = ", ")
        )
      }

      ingress = TorchIngressToken(
        features = task$feature_names,
        batchgetter = batchgetter,
        shape = private$.shape(task, param_vals)
      )

      self$state = list(ingress$shape)  # PipeOp API requires us to only set this to some list.
      list(ModelDescriptor(
        graph = graph,
        ingress = structure(list(ingress), names = graph$input$name),
        task = task,
        pointer = as.character(graph$output[, c("op.id", "channel.name"), with = FALSE]),
        pointer_shape = ingress$shape
      ))
    },
    .predict = function(inputs) inputs
  ),
  active = list(
    #' @field feature_types (`character(1)`)\cr
    #'   The features types that can be consumed by this `PipeOpTorchIngress`.
    feature_types = function(rhs) {
      assert_ro_binding(rhs)
      private$.feature_types
    }
  )
)

#' @title Torch Ingress Token
#'
#' @description
#' This function creates an S3 class of class `"TorchIngressToken"`, which is an internal data structure.
#' It contains the (meta-)information of how a batch is generated from a [`Task`][mlr3::Task] and fed into an entry point
#' of the neural network. It is stored as the `ingress` field in a [`ModelDescriptor`].
#'
#' @param features (`character`)\cr
#'   Features on which the batchgetter will operate.
#' @param batchgetter (`function`)\cr
#'   Function with two arguments: `data` and `device`. This function is given
#'   the output of `Task$data(rows = batch_indices, cols = features)`
#'   and it should produce a tensor of shape `shape_out`.
#' @param shape (`integer`)\cr
#'   Shape that `batchgetter` will produce. Batch-dimension should be included as `NA`.
#' @return `TorchIngressToken` object.
#' @family Graph Network
#' @export
#' @examplesIf torch::torch_is_installed()
#' # Define a task for which we want to define an ingress token
#' task = tsk("iris")
#'
#' # We create an ingress token for two feature Sepal.Length and Petal.Length:
#' # We have to specify the features, the batchgetter and the shape
#' features = c("Sepal.Length", "Petal.Length")
#' # As a batchgetter we use batchgetter_num
#'
#' batch_dt = task$data(rows = 1:10, cols =features)
#' batch_dt
#' batch_tensor = batchgetter_num(batch_dt, "cpu")
#' batch_tensor
#'
#' # The shape is unknown in the first dimension (batch dimension)
#'
#' ingress_token = TorchIngressToken(
#'   features = features,
#'   batchgetter = batchgetter_num,
#'   shape = c(NA, 2)
#' )
#' ingress_token
#'
TorchIngressToken = function(features, batchgetter, shape) {
  assert_character(features, any.missing = FALSE)
  assert_function(batchgetter, args = c("data", "device"))
  assert_integerish(shape)
  structure(list(
    features = features,
    batchgetter = batchgetter,
    shape = shape
  ), class = "TorchIngressToken")
}

#' @export
#' @keywords internal
hash_input.TorchIngressToken = function(x) {
  list(x$features, x$shape, hash_input(x$batchgetter))
}

#' @export
print.TorchIngressToken = function(x, ...) {
  cat(sprintf("Ingress: Task[%s] --> Tensor(%s)\n", str_collapse(x$features, n = 3, sep = ","), str_collapse(x$shape)))
}


#' @title Torch Entry Point for Numeric Features
#' @name mlr_pipeops_torch_ingress_num
#'
#' @description
#' Ingress PipeOp that represents a numeric (`integer()` and `numeric()`) entry point to a torch network.
#'
#' @inheritSection mlr_pipeops_torch_ingress Input and Output Channels
#' @inheritSection mlr_pipeops_torch_ingress State
#' @section Internals:
#' Uses [batchgetter_num()].
#'
#' @export
#' @family Graph Network
#' @family PipeOps
#' @examplesIf torch::torch_is_installed()
# We select the numeric features first
#' graph = po("select", selector = selector_type(c("numeric", "integer"))) %>>%
#'   po("torch_ingress_num")
#' task = tsk("german_credit")
#' # The output is a model descriptor
#' md = graph$train(task)[[1L]]
#' ingress = md$ingress[[1L]]
#' ingress$batchgetter(task$data(1:5, ingress$features), "cpu")
PipeOpTorchIngressNumeric = R6Class("PipeOpTorchIngressNumeric",
  inherit = PipeOpTorchIngress,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "torch_ingress_num", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, feature_types = c("numeric", "integer"))
    }
  ),
  private = list(
    .shape = function(task, param_vals) {
      c(NA, length(task$feature_types$type))
    },

    .get_batchgetter = function(task, param_vals) {
      batchgetter_num
    }
  )
)

#' @include zzz.R
register_po("torch_ingress_num", PipeOpTorchIngressNumeric)

#' @title Torch Entry Point for Categorical Features
#' @name mlr_pipeops_torch_ingress_categ
#'
#' @description
#' Ingress PipeOp that represents a categorical (`factor()`, `ordered()` and `logical()`) entry point to a torch network.
#'
#' @inheritSection mlr_pipeops_torch_ingress Input and Output Channels
#' @inheritSection mlr_pipeops_torch_ingress State
#' @section Parameters:
#' * `select` :: `logical(1)`\cr
#'   Whether `PipeOp` should selected the supported feature types. Otherwise it will err on receiving tasks
#'   with unsupported feature types.
#' @section Internals:
#' Uses [`batchgetter_categ()`].
#' @family PipeOps
#' @family Graph Network
#' @export
#' @examplesIf torch::torch_is_installed()
# We first select the categorical features
#' graph = po("select", selector = selector_type("factor")) %>>%
#'   po("torch_ingress_categ")
#' task = tsk("german_credit")
#' # The output is a model descriptor
#' md = graph$train(task)[[1L]]
#' ingress = md$ingress[[1L]]
#' ingress$batchgetter(task$data(1, ingress$features), "cpu")
PipeOpTorchIngressCategorical = R6Class("PipeOpTorchIngressCategorical",
  inherit = PipeOpTorchIngress,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "torch_ingress_categ", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, feature_types = c("factor", "ordered", "logical"))
    }
  ),
  private = list(
    .shape = function(task, param_vals) {
      # TODO: check that task is legit
      c(NA, length(task$feature_names))
    },
    .get_batchgetter = function(task, param_vals) {
      batchgetter_categ
    }
  )
)

register_po("torch_ingress_categ", PipeOpTorchIngressCategorical)

#' @title Ingress for Lazy Tensor
#' @name mlr_pipeops_torch_ingress_ltnsr
#' @description
#' Ingress for a single [`lazy_tensor`] column.
#'
#' @inheritSection mlr_pipeops_torch_ingress Input and Output Channels
#' @inheritSection mlr_pipeops_torch_ingress State
#'
#' @section Parameters:
#' * `shape` :: `integer()`\cr
#'   The shape of the tensor, where the first dimension (batch) must be `NA`.
#'   When it is not specified, the lazy tensor input column needs to have a known shape.
#'
#' @section Internals:
#' The returned batchgetter materializes the lazy tensor column to a tensor.
#' @family PipeOps
#' @family Graph Network
#' @export
#' @include utils.R shape.R
#' @examplesIf torch::torch_is_installed()
#' po_ingress = po("torch_ingress_ltnsr")
#' task = tsk("lazy_iris")
#'
#' md = po_ingress$train(list(task))[[1L]]
#'
#' ingress = md$ingress
#' x_batch = ingress[[1L]]$batchgetter(data = task$data(1, "x"), device = "cpu", cache = NULL)
#' x_batch
#'
#' # Now we try a lazy tensor with unknown shape, i.e. the shapes between the rows can differ
#'
#' ds = dataset(
#'   initialize = function() self$x = list(torch_randn(3, 10, 10), torch_randn(3, 8, 8)),
#'   .getitem = function(i) list(x = self$x[[i]]),
#'   .length = function() 2)()
#'
#' task_unknown = as_task_regr(data.table(
#'   x = as_lazy_tensor(ds, dataset_shapes = list(x = NULL)),
#'   y = rnorm(2)
#' ), target = "y", id = "example2")
#'
#' # this task (as it is) can NOT be processed by PipeOpTorchIngressLazyTensor
#' # It therefore needs to be preprocessed
#' po_resize = po("trafo_resize", size = c(6, 6))
#' task_unknown_resize = po_resize$train(list(task_unknown))[[1L]]
#'
#' # printing the transformed column still shows unknown shapes,
#' # because the preprocessing pipeop cannot infer them,
#' # however we know that the shape is now (3, 10, 10) for all rows
#' task_unknown_resize$data(1:2, "x")
#' po_ingress$param_set$set_values(shape = c(NA, 3, 6, 6))
#'
#' md2 = po_ingress$train(list(task_unknown_resize))[[1L]]
#'
#' ingress2 = md2$ingress
#' x_batch2 = ingress2[[1L]]$batchgetter(
#'   data = task_unknown_resize$data(1:2, "x"),
#'   device = "cpu",
#'   cache = NULL
#' )
#'
#' x_batch2
PipeOpTorchIngressLazyTensor = R6Class("PipeOpTorchIngressLazyTensor",
  inherit = PipeOpTorchIngress,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "torch_ingress_ltnsr", param_vals = list()) {
      param_set = ps(
        shape = p_uty(tags = "train", default = NULL, custom_check = crate(
          function(x) check_shape(x, null_ok = TRUE, unknown_batch = TRUE),
            .parent = topenv(), check_shape))
        )
      super$initialize(id = id, param_vals = param_vals, feature_types = "lazy_tensor", param_set = param_set)
    }
  ),
  private = list(
    .shape = function(task, param_vals) {
      lazy_cols = task$feature_types[get("type") == "lazy_tensor", "id"][[1L]]
      if (length(lazy_cols) != 1L) {
        stopf("PipeOpTorchIngressLazyTensor expects 1 'lazy_tensor' feature, but got %i.", length(lazy_cols))
      }
      example = task$data(task$row_ids[1L], lazy_cols)[[1L]]
      input_shape = dd(example)$pointer_shape
      pv_shape = param_vals$shape

      if (is.null(input_shape)) {
        if (is.null(pv_shape)) {
          stopf("If input shape is unknown, the 'shape' parameter must be set.")
        }
        return(pv_shape)
      }

      if (!is.null(pv_shape) && !isTRUE(all.equal(pv_shape, input_shape))) {
        stopf("Parameter 'shape' is set for PipeOp '%s', but differs from the (known) lazy tensor input shape.", self$id) # nolint
      }

      input_shape
    },
    .get_batchgetter = function(task, param_vals) {
      batchgetter_lazy_tensor
    }
  )
)

batchgetter_lazy_tensor = function(data, device, cache) {
  materialize_internal(x = data[[1L]], device = device, cache = cache, rbind = TRUE)
}

register_po("torch_ingress_ltnsr", PipeOpTorchIngressLazyTensor)
