#' @title Learner Torch Module
#'
#' @templateVar name module
#' @templateVar task_types classif, regr
#' @template params_learner
#' @template learner
#'
#' @description
#' Create a torch learner from a torch module.
#'
#' @template param_task_type
#' @param module_generator (`function` or `nn_module_generator`)\cr
#'   A `nn_module_generator` or `function` returning an `nn_module`.
#'   Both must take as argument the `task` for which to construct the network.
#'   Other arguments to its initialize method can be provided as parameters.
#' @param param_set (`NULL` or [`ParamSet`][paradox::ParamSet])\cr
#'   If provided, contains the parameters for the module_generator.
#'   If `NULL`, parameters will be inferred from the module_generator.
#' @param ingress_tokens (`list` of [`TorchIngressToken()`])\cr
#'   A list with ingress tokens that defines how the dataloader will be defined.
#'   The names must correspond to the arguments of the network's forward method.
#'   For numeric, categorical, and lazy tensor features, you can use [`ingress_num()`],
#'   [`ingress_categ()`], and [`ingress_ltnsr()`] to create them.
#' @template param_packages
#' @param feature_types (`NULL` or `character()`)\cr
#'   The feature types. Defaults to all available feature types.
#' @param properties (`NULL` or `character()`)\cr
#'   The properties of the learner.
#'   Defaults to all available properties for the given task type.
# @section Parameters: See [`LearnerTorch`] and constructor argument `param_set`.
#' @family Learner
#' @include LearnerTorch.R
#' @export
#' @examplesIf torch::torch_is_installed()
#' nn_one_layer = nn_module("nn_one_layer",
#'   initialize = function(task, size_hidden) {
#'     self$first = nn_linear(task$n_features, size_hidden)
#'     self$second = nn_linear(size_hidden, length(task$class_names))
#'   },
#'   # argument x corresponds to the ingress token x
#'   forward = function(x) {
#'     x = self$first(x)
#'     x = nnf_relu(x)
#'     self$second(x)
#'   }
#' )
#' learner = lrn("classif.module",
#'   module_generator = nn_one_layer,
#'   ingress_tokens = list(x = ingress_num()),
#'   epochs = 10,
#'   size_hidden = 20,
#'   batch_size = 16
#' )
#' task = tsk("iris")
#' learner$train(task)
#' learner$network
LearnerTorchModule = R6Class("LearnerTorchModule",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(module_generator = NULL, param_set = NULL, ingress_tokens = NULL,
      task_type, properties = NULL, optimizer = NULL, loss = NULL, callbacks = list(),
      packages = character(0), feature_types = NULL) {
      assert(check_class(module_generator, "nn_module_generator"), check_function(module_generator))
      private$.module_generator = module_generator
      args = names(formals(module_generator))
      if (!"task" %in% args) {
        stopf("module_generator must have 'task' as a parameter")
      }

      private$.ingress_tokens = assert_list(ingress_tokens, types = "TorchIngressToken", names = "unique", min.len = 1L)

      if (is.null(feature_types)) {
        feature_types = unname(mlr_reflections$task_feature_types)
      } else {
        assert_subset(feature_types, mlr_reflections$task_feature_types)
      }

      if (is.null(properties)) {
        properties = mlr_reflections$learner_properties[[task_type]]
      } else {
        properties = assert_subset(properties, mlr_reflections$learner_properties[[task_type]])
      }

      # If param_set is NULL, try to infer parameters from module_generator using inferps
      if (is.null(param_set)) {
        param_set = inferps(module_generator, ignore = "task", tags = "train")
      }

      super$initialize(
        id = paste0(task_type, ".module"),
        loss = loss,
        optimizer = optimizer,
        callbacks = callbacks,
        task_type = task_type,
        label = "Custom Module",
        properties = properties,
        packages = packages,
        param_set = param_set,
        feature_types = feature_types,
        man = "mlr3torch::mlr_learners.module"
      )
    }
  ),
  private = list(
    .ingress_tokens = NULL,
    .module_generator = NULL,

    .network = function(task, param_vals) {
      module_params = param_vals[names(param_vals) %in% formalArgs(private$.module_generator)]
      invoke(private$.module_generator, task = task, .args = module_params)
    },

    .dataset = function(task, param_vals) {
      ingress_tokens = private$.ingress_tokens
      dataset = task_dataset(
        task,
        feature_ingress_tokens = ingress_tokens,
        target_batchgetter = get_target_batchgetter(self$task_type)
      )
    },

    .additional_phash_input = function() {
      list(self$properties, self$feature_types, private$.module_generator, self$packages, private$.ingress_tokens)
    }
  )
)

nn_placeholder = nn_module("nn_placeholder",
  initialize = function(task) NULL,
  forward = function(x) {
    x
  }
)

#' @include PipeOpTorchIngress.R task_dataset.R
register_learner("classif.module", LearnerTorchModule, module_generator = nn_placeholder, ingress_tokens = list(x = ingress_num()))
register_learner("regr.module", LearnerTorchModule, module_generator = nn_placeholder, ingress_tokens = list(x = ingress_num()))
