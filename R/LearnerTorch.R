#' @title Base Class for Torch Learners
#'
#' @name mlr_learners_torch
#'
#' @description
#' This base class provides the basic functionality for training and prediction of a neural network.
#' All torch learners should inherit from this class.
#'
#' It also allows to hook into the training loop via a callback mechanism.
#'
#' @template param_id
#' @template param_task_type
#' @template param_param_vals
#' @template param_param_set
#' @template param_properties
#' @template param_packages
#' @template param_feature_types
#' @template param_man
#' @template param_label
#' @param predict_types (`character()`)\cr
#'   The predict types.
#'   See [`mlr_reflections$learner_predict_types`][mlr_reflections] for available values.
#'   For regression, the default is `"response"`.
#'   For classification, this defaults to `"response"` and `"prob"`.
#'   To deviate from the defaults, it is necessary to overwrite the private `$.encode_prediction()`
#'   method, see section *Inheriting*.
#' @param loss (`NULL` or [`TorchLoss`])\cr
#'   The loss to use for training.
#'   Defaults to MSE for regression and cross entropy for classification.
#' @param optimizer (`NULL` or [`TorchOptimizer`])\cr
#'   The optimizer to use for training.
#'   Defaults to adam.
#' @param callbacks (`list()` of [`TorchCallback`]s)\cr
#'   The callbacks to use for training.
#'   Defaults to an empty` list()`, i.e. no callbacks.
#'
#' @section Model:
#' The Model is a list with elements `network`, `loss_state`, `optimizer_state`, `callbacks` and `seed`.
#'
#' @template paramset_torchlearner
#'
#' @section Inheriting:
#' There are no seperate classes for classification and regression to inherit from.
#' Instead, the `task_type` must be specified  as a construction argument.
#' Currently, only classification and regression are supported.
#'
#' When inheriting from this class, one should overload two private methods:
#'
#' * `.network(task, param_vals)`\cr
#'   ([`Task`], `list()`) -> [`nn_module`]\cr
#'   Construct a [`torch::nn_module`] object for the given task and parameter values, i.e. the neural network that
#'   is trained by the learner.
#'   For classification, the output of this network are expected to be the scores before the application of the
#'   final softmax layer.
#' * `.dataset(task, param_vals)`\cr
#'   ([`Task`], `list()`) -> [`torch::dataset`]\cr
#'   Create the dataset for the task.
#'   Must respect the parameter value of the device.
#'   Moreover, one needs to pay attention respect the row ids of the provided task.
#'
#' It is also possible to overwrite the private `.dataloader()` method instead of the `.dataset()` method.
#' Per default, a dataloader is constructed using the output from the `.dataset()` method.
#'
#' * `.dataloader(task, param_vals)`\cr
#'   ([`Task`], `list()`) -> [`torch::dataloader`]\cr
#'   Create a dataloader from the task.
#'   Needs to respect at least `batch_size` and `shuffle` (otherwise predictions can be permuted).
#'
#' To change the predict types, the private `.encode_prediction()` method can be overwritten:
#'
#' * `.encode_prediction(predict_tensor, task, param_vals)`\cr
#'   ([`torch_tensor`], [`Task`], `list()`) -> `list()`\cr
#'   Take in the raw predictions from `self$network` (`predict_tensor`) and encode them into a
#'   format that can be converted to valid `mlr3` predictions using [`mlr3::as_prediction_data()`].
#'   This method must take `self$predict_type` into account.
#'
#' While it is possible to add parameters by specifying the `param_set` construction argument, it is currently
#' not possible to remove existing parameters, i.e. those listed in section *Parameters*.
#' None of the parameters provided in `param_set` can have an id that starts with `"loss."`, `"opt.",
#' or `"cb."`, as these are preserved for the dynamically constructed parameters of the optimizer, the loss function,
#' and the callbacks.
#'
#' To perform additional input checks on the task, the private `.verify_train_task(task, param_vals)` and
#' `.verify_predict_task(task, param_vals)` can be overwritten.
#'
#' @family Learner
#' @export
LearnerTorch = R6Class("LearnerTorch",
  inherit = Learner,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, task_type, param_set, properties, man, label, feature_types,
      optimizer = NULL, loss = NULL, packages = NULL, predict_types = NULL, callbacks = list()) {

      assert_choice(task_type, c("regr", "classif"))
      predict_types = predict_types %??% switch(task_type,
        regr = "response",
        classif = c("response", "prob")
      )
      if (is.null(loss)) {
        private$.loss = t_loss(switch(task_type, classif = "cross_entropy", regr = "mse"))
      } else {
        private$.loss = as_torch_loss(loss, clone = TRUE)
      }

      if (task_type %nin% private$.loss$task_types) {
        stopf("Loss only supports task types %s, but learner has type \"%s\".",
          paste0("\"", private$.loss$task_types, "\"", sep = ", "), task_type
        )
      }

      if (is.null(optimizer)) {
        private$.optimizer = t_opt("adam")
      } else {
        private$.optimizer = as_torch_optimizer(optimizer, clone = TRUE)
      }

      private$.optimizer$param_set$set_id = "opt"
      private$.loss$param_set$set_id = "loss"
      callbacks = as_torch_callbacks(callbacks, clone = TRUE)
      callback_ids = ids(callbacks)
      if (!test_names(callback_ids, type = "unique")) {
        stopf("All callbacks must have unique IDs that are valid names, but they are %s.",
          paste0("'", callback_ids, "'", collapse = ", ")
        )
      }

      private$.callbacks = set_names(callbacks, callback_ids)
      walk(private$.callbacks, function(cb) {
        cb$param_set$set_id = paste0("cb.", cb$id)
      })

      packages = unique(c(
        packages,
        unlist(map(private$.callbacks, "packages")),
        private$.loss$packages,
        private$.optimizer$packages
      ))

      assert_subset(properties, mlr_reflections$learner_properties[[task_type]])
      assert_subset(predict_types, names(mlr_reflections$learner_predict_types[[task_type]]))
      if (any(grepl("^(loss\\.|opt\\.|cb\\.)", param_set$ids()))) {
        stopf("Prefixes 'loss.', 'opt.', and 'cb.' are reserved for dynamically constructed parameters.")
      }
      packages = assert_character(packages, any.missing = FALSE, min.chars = 1L)
      packages = union(c("mlr3", "mlr3torch"), packages)

      paramset_torch = paramset_torchlearner(task_type)
      if (param_set$length > 0) {
        private$.param_set_base = ParamSetCollection$new(list(param_set, paramset_torch))
      } else {
        private$.param_set_base = paramset_torch
      }

      # explanation of the self$param_set call:
      # As of now, private$.param_set is NULL, this will cause the ParamSetCollection to be constructed
      # (as self$param_set) is an active binding.
      # However we then pass this constructed paramset to the learner parent class, which will assign it to self$param_set
      # However this behind the scene will once again set it to private$.param_set as it causes the function in
      # self$param_set with an rhs to be called, which in turn assigns it (again) to private$.param_set
      super$initialize(
        id = id,
        task_type = task_type,
        packages = packages,
        param_set = self$param_set,
        predict_types = predict_types,
        properties = properties,
        data_formats = "data.table",
        label = label,
        feature_types = feature_types,
        man = man
      )
    },
    #' @description
    #' Helper for print outputs.
    #' @param ... (ignored).
    format = function(...) {
      sprintf("<%s[%s]:%s>", class(self)[1L], self$task_type, self$id)
    },

    #' @description
    #' Prints the object.
    #' @param ... (any)\cr
    #'   Currently unused.
    print = function(...) {
      super$print(...)
      catn(str_indent("* Optimizer:", private$.optimizer$id))
      catn(str_indent("* Loss:", private$.loss$id))
      catn(str_indent("* Callbacks:", if (length(private$.callbacks)) as_short_string(paste0(ids(private$.callbacks), collapse = ","), 1000L) else "-"))
    }
  ),
  active = list(
    #' @field network ([`nn_module()`][torch::nn_module])\cr
    #' Shortcut for `learner$model$network`.
    network = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$network
    },
    #' @field param_set ([`ParamSet`])\cr
    #'   The parameter set
    param_set = function(rhs) {
      if (is.null(private$.param_set)) {
        private$.param_set = ParamSetCollection$new(c(
          list(private$.param_set_base, private$.optimizer$param_set, private$.loss$param_set),
          map(private$.callbacks, "param_set"))
        )
      }
      private$.param_set
    },
    #' @field callbacks ([`CallbackSetHistory`])\cr
    #' Shortcut for `learner$model$callbacks`.
    callbacks = function(rhs) {
      assert_ro_binding(rhs)
      self$model$callbacks
    }
    # hash = function(rhs) {
    #   assert_ro_binding(rhs)
    #   calculate_hash(super$hash, )
    #
    # },
    # phash = function(rhs) {
    #   assert_ro_binding(rhs)
    #   calclate_hash(super$hash,
    #     private$.optimizer$phash,
    #     private$.loss$phash,
    #     map(private$.callbacks, "phash")
    #   )
    # }
  ),
  private = list(
    .train = function(task) {
      param_vals = self$param_set$get_values(tags = "train")
      first_row = task$head(1)
      measures = c(normalize_to_list(param_vals$measures_train), normalize_to_list(param_vals$measures_valid))
      available_predict_types = mlr_reflections$learner_predict_types[[self$task_type]][[self$predict_type]]
      walk(measures, function(m) {
        if (m$predict_type %nin% available_predict_types) {
          stopf(paste0("Measure '%s' requires predict type '%s' but learner has '%s'.\n",
              "Change the predict type or select other measures."),
            m$id, m$predict_type, self$predict_type)
        }
      })

      iwalk(first_row, function(x, nm) {
        if (!is_lazy_tensor(x)) return(NULL)
        predict_shape = dd(x)$pointer_shape_predict
        train_shape = dd(x)$pointer_shape
        if (is.null(train_shape) || is.null(predict_shape)) {
          return(NULL)
        }
        if (!isTRUE(all.equal(train_shape, predict_shape))) {
          stopf("Lazy tensor column '%s' has a different shape during training (%s) and prediction (%s).",
            nm, paste0(train_shape, collapse = "x"), paste0(predict_shape, collapse = "x"))
        }
      })
      private$.verify_train_task(task, param_vals)

      param_vals$device = auto_device(param_vals$device)
      if (param_vals$seed == "random") param_vals$seed = sample.int(10000000L, 1L)

      model = with_torch_settings(seed = param_vals$seed, num_threads = param_vals$num_threads, {
        learner_torch_train(self, private, super, task, param_vals)
      })
      model$task_col_info = copy(task$col_info)
      return(model)
    },
    .predict = function(task) {
      cols = c(task$feature_names, task$target_names)
      ci_predict = task$col_info[get("id") %in% cols, c("id", "type", "levels")]
      ci_train = self$model$task_col_info[get("id") %in% cols, c("id", "type", "levels")]
      # permuted factor levels cause issues, because we are converting fct -> int
      if (!test_equal_col_info(ci_train, ci_predict)) { # nolint
        stopf(paste0(
          "Predict task's column info does not match the train task's column info.\n",
          "This migth be handled more gracefully in the future.\n",
          "Training column info:\n'%s'\n",
          "Prediction column info:\n'%s'"),
          paste0(capture.output(ci_train), collapse = "\n"),
          paste0(capture.output(ci_predict), collapse = "\n"))
      }
      param_vals = self$param_set$get_values(tags = "predict")
      private$.verify_predict_task(task, param_vals)
      # FIXME: https://github.com/mlr-org/mlr3/issues/946
      # This addresses the issues with the factor levels and is only a temporary fix
      # Should be handled outside of mlr3torch
      # Ideally we could rely on state$train_task, but there is this complication
      # https://github.com/mlr-org/mlr3/issues/947
      param_vals$device = auto_device(param_vals$device)

      with_torch_settings(seed = self$model$seed, num_threads = param_vals$num_threads, {
        learner_torch_predict(self, private, super, task, param_vals)
      })
    },
    .encode_prediction = function(predict_tensor, task) {
      encode_prediction_default(
        predict_tensor = predict_tensor,
        predict_type = self$predict_type,
        task = task
      )
    },
    .network = function(task, param_vals) stop(".network must be implemented."),
    # the dataloader gets param_vals that may be different from self$param_set$values, e.g.
    # when the dataloader for validation data is loaded, `shuffle` is set to FALSE.
   .dataloader = function(task, param_vals) {
      dataloader(
        dataset = private$.dataset(task, param_vals),
        batch_size = param_vals$batch_size,
        shuffle = param_vals$shuffle,
        drop_last = param_vals$drop_last
      )
    },
    .dataloader_predict = function(task, param_vals) {
      param_vals_test = insert_named(param_vals, list(shuffle = FALSE, drop_last = FALSE))
      private$.dataloader(task, param_vals_test)
    },
    .dataset = function(task, param_vals) {
      task_dataset(
        task = task,
        feature_ingress_tokens = private$.feature_ingress_tokens(task, param_vals),
        target_batchgetter = private$.target_batchgetter(task, param_vals),
        device = param_vals$device
      )

    },
    .feature_ingress_tokens = function(task, param_vals) {

    },
    .target_batchgetter = function(task, param_vals) {
      get_target_batchgetter(task$task_type)
    },
    .optimizer = NULL,
    .loss = NULL,
    .param_set_base = NULL,
    .callbacks = NULL,
    .verify_train_task = function(task, param_vals) NULL,
    .verify_predict_task = function(task, param_vals) NULL,
    deep_clone = function(name, value) {
      private$.param_set = NULL # required to keep clone identical to original, otherwise tests get really ugly
      # FIXME this repairs the mlr3::Learner deep_clone() method which is broken.
      if (is.environment(value) && !is.null(value[[".__enclos_env__"]])) {
        return(value$clone(deep = TRUE))
      } else if (test_class(value, "nn_module")) {
        value$clone(deep = TRUE)
      } else if (name == ".callbacks") {
        map(value, function(x) x$clone(deep = TRUE))
      } else if (name == "state") {
        if (!is.null(value)) {
          model = value$model
          value["model"] = list(NULL)
          value = super$deep_clone(name, value)
          value[["model"]] = list(
            network = model$network$clone(deep = TRUE),
            loss_state = clone_recurse(model$loss_state),
            optimizer_state = clone_recurse(model$optimizer_state),
            callbacks = map(model$callbacks, function(x) x$clone(deep = TRUE)),
            seed = model$seed,
            task_col_info = copy(model$task_col_info)
          )
        }
        return(value)
      } else if (name == ".param_set") {
        NULL
      } else {
        super$deep_clone(name, value)
      }
    }
  )
)

clone_recurse = function(l) {
  if (test_class(l, "torch_tensor")) {
    return(l$clone())
  } else if (test_list(l) && length(l) > 0L) {
    map(l, clone_recurse)
  } else {
    return(l)
  }
}
