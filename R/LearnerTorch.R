#' @title Base Class for Torch Learners
#'
#' @name mlr_learners_torch
#'
#' @description
#' This base class provides the basic functionality for training and prediction of a neural network.
#' All torch learners should inherit from this class.
#'
#' @section Validation:
#' To specify the validation data, you can set the `$validate` field of the Learner, which can be set to:
#' * `NULL`: no validation
#' * `ratio`: only proportion `1 - ratio` of the task is used for training and `ratio` is used for validation.
#' * `"test"` means that the `"test"` task of a resampling is used and is not possible when calling `$train()` manually.
#' * `"predefined"`: This will use the predefined `$internal_valid_task` of a [`mlr3::Task`], which can e.g.
#'   be created using the `$divide()` method  of `Task`.
#'
#' This validation data can also be used for early stopping, see the description of the `Learner`'s parameters.
#'
#' @section Saving a Learner:
#' In order to save a `LearnerTorch` for later usage, it is necessary to call the `$marshal()` method on the `Learner`
#' before writing it to disk, as the object will otherwise not be saved correctly.
#' After loading a marshaled `LearnerTorch` into R again, you then need to call `$unmarshal()` to transform it
#' into a useable state.
#'
#' @section Early Stopping and Tuning:
#' In order to prevent overfitting, the `LearnerTorch` class allows to use early stopping via the `patience`
#' and `min_delta` parameters, see the `Learner`'s parameters.
#' When tuning a `LearnerTorch` it is also possible to combine the explicit tuning via `mlr3tuning`
#' and the `LearnerTorch`'s internal tuning of the epochs via early stopping.
#' To do so, you just need to include `epochs = to_tune(upper = <upper>, internal = TRUE)` in the search space,
#' where `<upper>` is the maximally allowed number of epochs, and configure the early stopping.
#'
#' @template param_id
#' @template param_task_type
#' @template param_param_vals
#' @template param_properties
#' @template param_packages
#' @template param_feature_types
#' @template param_man
#' @template param_label
#' @param param_set ([`ParamSet`][paradox::ParamSet] or `alist()`)\cr
#'   Either a parameter set, or an `alist()` containing different values of self,
#'   e.g. `alist(private$.param_set1, private$.param_set2)`, from which a [`ParamSet`][paradox::ParamSet] collection
#'   should be created.
#' @param predict_types (`character()`)\cr
#'   The predict types.
#'   See [`mlr_reflections$learner_predict_types`][mlr3::mlr_reflections] for available values.
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
#' The Model is a list of class `"learner_torch_model"` with the following elements:
#'   * `network` :: The trained [network][torch::nn_module].
#'   * `optimizer` :: The `$state_dict()` [optimizer][torch::optimizer] used to train the network.
#'   * `loss_fn` :: The `$state_dict()` of the [loss][torch::nn_module] used to train the network.
#'   * `callbacks` :: The [callbacks][mlr3torch::mlr_callback_set] used to train the network.
#'   * `seed` :: The seed that was / is used for training and prediction.
#'   * `epochs` :: How many epochs the model was trained for (early stopping).
#'   * `task_col_info` :: A `data.table()` containing information about the train-task.
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
#'   ([`Task`][mlr3::Task], `list()`) -> [`nn_module`][torch::nn_module]\cr
#'   Construct a [`torch::nn_module`] object for the given task and parameter values, i.e. the neural network that
#'   is trained by the learner.
#'   For classification, the output of this network are expected to be the scores before the application of the
#'   final softmax layer.
#' * `.dataset(task, param_vals)`\cr
#'   ([`Task`][mlr3::Task], `list()`) -> [`torch::dataset`]\cr
#'   Create the dataset for the task.
#'   Must respect the parameter value of the device.
#'   Moreover, one needs to pay attention respect the row ids of the provided task.
#'
#' It is also possible to overwrite the private `.dataloader()` method instead of the `.dataset()` method.
#' Per default, a dataloader is constructed using the output from the `.dataset()` method.
#' However, this should respect the dataloader parameters from the [`ParamSet`][paradox::ParamSet].
#'
#' * `.dataloader(task, param_vals)`\cr
#'   ([`Task`][mlr3::Task], `list()`) -> [`torch::dataloader`]\cr
#'   Create a dataloader from the task.
#'   Needs to respect at least `batch_size` and `shuffle` (otherwise predictions can be permuted).
#'
#' To change the predict types, the private `.encode_prediction()` method can be overwritten:
#'
#' * `.encode_prediction(predict_tensor, task, param_vals)`\cr
#'   ([`torch_tensor`][torch::torch_tensor], [`Task`][mlr3::Task], `list()`) -> `list()`\cr
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
#' For learners that have other construction arguments that should change the hash of a learner, it is required
#' to implement the private `$.additional_phash_input()`.
#'
#' @family Learner
#' @export
LearnerTorch = R6Class("LearnerTorch",
  inherit = Learner,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, task_type, param_set, properties, man, label, feature_types,
      optimizer = NULL, loss = NULL, packages = character(), predict_types = NULL, callbacks = list()) {
      assert_choice(task_type, c("regr", "classif"))

      predict_types = predict_types %??% switch(task_type,
        regr = "response",
        classif = c("response", "prob")
      )

      assert_subset(properties, mlr_reflections$learner_properties[[task_type]])
      properties = union(properties, c("marshal", "validation", "internal_tuning"))
      assert_subset(predict_types, names(mlr_reflections$learner_predict_types[[task_type]]))
      packages = assert_character(packages, any.missing = FALSE, min.chars = 1L)
      packages = union(c("mlr3", "mlr3torch"), packages)

      private$.param_set_torch = paramset_torchlearner(task_type)

      check_ps = function(param_set) {
        assert_param_set(param_set)
        if (any(grepl("^(loss\\.|opt\\.|cb\\.)", param_set$ids()))) {
          stopf("Prefixes 'loss.', 'opt.', and 'cb.' are reserved for dynamically constructed parameters.")
        }
      }

      if (test_class(param_set, "ParamSet")) {
        check_ps(param_set)
        if (!is.null(private$.param_set_base)) {
          stopf("Learner '%s': Don't set .param_set_base before passing a ParamSet to param_set", self$id)
        }
        private$.param_set_base = param_set
        private$.param_set_source = alist(private$.param_set_base)
      } else {
        lapply(param_set, function(x) {
          # otherwise cloning can fail when parameter values are set in the param_set constructed
          # from expressions in alist()
          assert_true(grepl("^(self|private|super)", deparse(x)))
          check_ps(eval(x))
        })
        private$.param_set_source = param_set
      }
      # loss needs access to the task_type
      self$task_type = task_type
      if (is.null(loss)) {
        private$.loss = t_loss(switch(task_type, classif = "cross_entropy", regr = "mse"))
      } else if (!inherits(loss, "LossNone")) {
        self$loss = loss
      }

      if (is.null(optimizer)) {
        private$.optimizer = t_opt("adam")
      } else if (!inherits(optimizer, "OptimizerNone")) {
        self$optimizer = optimizer
      }

      if (!inherits(callbacks, "CallbacksNone")) {
        self$callbacks = callbacks
      }

      packages = unique(c(
        packages,
        unlist(map(private$.callbacks, "packages")),
        private$.loss$packages,
        private$.optimizer$packages))

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
    },
    #' @description
    #' Marshal the learner.
    #' @param ... (any)\cr
    #'   Additional parameters.
    #' @return self
    marshal = function(...) {
      learner_marshal(.learner = self, ...)
    },
    #' @description
    #' Unmarshal the learner.
    #' @param ... (any)\cr
    #'   Additional parameters.
    #' @return self
    unmarshal = function(...) {
      learner_unmarshal(.learner = self, ...)
    },
    #' @description
    #' Create the dataset for a task.
    #' @param task [`Task`][mlr3::Task]\cr
    #' The task
    #' @return [`dataset`][torch::dataset]
    dataset = function(task) {
      assert_task(task)
      param_vals = self$param_set$values
      param_vals$device = auto_device(param_vals$device)

      private$.dataset(task, param_vals)
    }
  ),
  active = list(
    #' @field validate
    #' How to construct the internal validation data. This parameter can be either `NULL`,
    #' a ratio in $(0, 1)$, `"test"`, or `"predefined"`.
    validate = function(rhs) {
      if (!missing(rhs)) {
        private$.validate = assert_validate(rhs)
      }
      private$.validate
    },

    #' @field loss ([`TorchLoss`])\cr
    #' The torch loss.
    loss = function(rhs) {
      if (!missing(rhs)) {
        private$.param_set = NULL
        loss = as_torch_loss(rhs, clone = TRUE)
        assert_choice(self$task_type, loss$task_types)
        private$.loss = loss
        self$packages = unique(c(self$packages, private$.loss$packages))
      }
      private$.loss
    },

    #' @field optimizer ([`TorchOptimizer`])\cr
    #' The torch optimizer.
    optimizer = function(rhs) {
      if (!missing(rhs)) {
        private$.optimizer = as_torch_optimizer(rhs, clone = TRUE)
        private$.param_set = NULL
        self$packages = unique(c(self$packages, private$.optimizer$packages))
      }
      private$.optimizer
    },

    #' @field callbacks (`list()` of [`TorchCallback`]s)\cr
    #' List of torch callbacks.
    #' The ids will be set as the names.
    callbacks = function(rhs) {
      if (!missing(rhs)) {
        callbacks = as_torch_callbacks(rhs, clone = TRUE)
        callback_ids = ids(callbacks)
        if (!test_names(callback_ids, type = "unique")) {
          stopf("All callbacks must have unique IDs that are valid names, but they are %s.",
            paste0("'", callback_ids, "'", collapse = ", ")
          )
        }
        private$.callbacks = callbacks
        private$.param_set = NULL
        self$packages = unique(c(self$packages, unlist(map(private$.callbacks, "packages"))))
      }
      private$.callbacks
    },

    #' @field internal_valid_scores
    #' Retrieves the internal validation scores as a named `list()`.
    #' Specify the `$validate` field and the `measures_valid` parameter to configure this.
    #' Returns `NULL` if learner is not trained yet.
    internal_valid_scores = function() {
      self$state$internal_valid_scores
    },
    #' @field internal_tuned_values
    #' When early stopping is activate, this returns a named list with the early-stopped epochs,
    #' otherwise an empty list is returned.
    #' Returns `NULL` if learner is not trained yet.
    internal_tuned_values = function() {
      self$state$internal_tuned_values
    },
    #' @field marshaled (`logical(1)`)\cr
    #' Whether the learner is marshaled.
    marshaled = function(rhs) {
      assert_ro_binding(rhs)
      learner_marshaled(self)
    },
    #' @field network ([`nn_module()`][torch::nn_module])\cr
    #' Shortcut for `learner$model$network`.
    network = function(rhs) {
      assert_ro_binding(rhs)
      self$state$model$network
    },
    #' @field param_set ([`ParamSet`][paradox::ParamSet])\cr
    #'   The parameter set
    param_set = function(rhs) {
      if (is.null(private$.param_set)) {
        # optimizer, loss and callbacks don't have to be part of the param_set, they can also be
        # parameters themselves
        sourcelist = lapply(private$.param_set_source, function(x) eval(x))
        private$.param_set = ParamSetCollection$new(c(
          list(private$.param_set_torch),
          sourcelist,
          if (!is.null(private$.optimizer)) list(opt = private$.optimizer$param_set),
          if (!is.null(private$.loss)) list(loss = private$.loss$param_set),

          if (!is.null(private$.callbacks)) {
            set_names(map(private$.callbacks, "param_set"), sprintf("cb.%s", ids(private$.callbacks)))
          }
        ))
      }
      if (!missing(rhs) && !identical(rhs, private$.param_set)) {
        stopf("parameter set is read-only")
      }
      private$.param_set
    },
    #' @field hash (`character(1)`)\cr
    #' Hash (unique identifier) for this object.
    hash = function(rhs) {
       assert_ro_binding(rhs)
       calculate_hash(c(list(self$phash), self$param_set$values))
    },
    #' @field phash (`character(1)`)\cr
    #' Hash (unique identifier) for this partial object, excluding some components
    #' which are varied systematically during tuning (parameter values).
    phash = function(rhs) {
      assert_ro_binding(rhs)
      calculate_hash(super$phash,
        self$task_type,
        private$.optimizer$phash,
        private$.loss$phash,
        map(private$.callbacks, "phash"),
        private$.additional_phash_input()
      )
    }
  ),
  private = list(
    .param_set_torch = NULL,
    .param_set_source = NULL,
    .param_set_base = NULL,
    .extract_internal_tuned_values = function() {
      if (self$state$param_vals$patience == 0) {
        named_list()
      } else {
        list(epochs = self$model$epochs)
      }
    },
    .extract_internal_valid_scores = function() {
      if (is.null(self$model$internal_valid_scores)) {
        named_list()
      } else {
        self$model$internal_valid_scores
      }
    },
    .validate = NULL,
    .additional_phash_input = function() {
      if (is.null(self$initialize)) return(NULL)
      initformals = names(formals(args(self$initialize)))
      if (!test_subset(initformals, c("task_type", "loss", "optimizer", "callbacks"))) {
        stopf("Learner %s has non-standard construction arguments, implement .additional_phash_input()",
        self$id)
      }
    },
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
      model$task_col_info = copy(task$col_info[c(task$feature_names, task$target_names), c("id", "type", "levels")])
      return(model)
    },
    .predict = function(task) {
      param_vals = self$param_set$get_values(tags = "predict")
      cols = c(task$feature_names, task$target_names)
      ci_predict = task$col_info[get("id") %in% cols, c("id", "type", "levels")]
      ci_train = self$model$task_col_info[get("id") %in% cols, c("id", "type", "levels")]
      # permuted factor levels cause issues, because we are converting fct -> int
      # FIXME: https://github.com/mlr-org/mlr3/issues/946
      # This addresses the issues with the factor levels and is only a temporary fix
      # Should be handled outside of mlr3torch
      # Ideally we could rely on state$train_task, but there is this complication
      # https://github.com/mlr-org/mlr3/issues/947
      param_vals$device = auto_device(param_vals$device)
      if (!test_equal_col_info(ci_train, ci_predict)) { # nolint
        stopf(paste0(
          "Predict task's column info does not match the train task's column info.\n",
          "This migth be handled more gracefully in the future.\n",
          "Training column info:\n'%s'\n",
          "Prediction column info:\n'%s'"),
          paste0(capture.output(ci_train), collapse = "\n"),
          paste0(capture.output(ci_predict), collapse = "\n"))
      }
      private$.verify_predict_task(task, param_vals)

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
      dl_args = c(
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "num_workers",
        "collate_fn",
        "pin_memory",
        "drop_last",
        "timeout",
        "worker_init_fn",
        "worker_globals",
        "worker_packages"
      )
      args = param_vals[names(param_vals) %in% dl_args]
      invoke(dataloader, dataset = private$.dataset(task, param_vals), .args = args)
    },
    .dataloader_predict = function(task, param_vals) {
      param_vals_test = insert_named(param_vals, list(shuffle = FALSE, drop_last = FALSE))
      private$.dataloader(task, param_vals_test)
    },
    .dataset = function(task, param_vals) {
      stopf(".dataset must be implememnted")
    },
    .optimizer = NULL,
    .loss = NULL,
    .callbacks = NULL,
    .verify_train_task = function(task, param_vals) NULL,
    .verify_predict_task = function(task, param_vals) NULL,
    deep_clone = function(name, value) {
      private$.param_set = NULL # required to keep clone identical to original, otherwise tests get really ugly
      if (is.R6(value)) {
        return(value$clone(deep = TRUE))
      } else if (test_class(value, "nn_module")) {
        value$clone(deep = TRUE)
      } else if (name == ".callbacks") {
        if (is.null(value)) return(NULL)
        map(value, function(x) x$clone(deep = TRUE))
      } else if (name == ".param_set") {
        NULL
      } else if (name == "state") {
        if (!is.null(value)) {
          model = value$model
          value["model"] = list(NULL)
          value = super$deep_clone(name, value)
          model$network = model$network$clone(deep = TRUE)
          model$loss_fn = clone_recurse(model$loss_fn)
          model$callbacks = map(model$callbacks, function(x) {
              if (is.R6(x)) {
                x$clone(deep = TRUE)
              } else {
                x
              }
          })
          value$model = model
        }
        return(value)
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

#' @export
marshal_model.learner_torch_model = function(model, inplace = FALSE, ...) {
  model$network = torch_serialize(model$network)
  model$loss_fn = torch_serialize(model$loss_fn)
  model$optimizer = torch_serialize(model$optimizer)

  structure(list(
    marshaled = model,
    packages = "mlr3torch"
  ), class = c("learner_torch_model_marshaled", "list_marshaled", "marshaled"))
}

#' @export
unmarshal_model.learner_torch_model_marshaled = function(model, inplace = FALSE, device = "cpu", ...) {
  model = model$marshaled
  model$network = torch_load(model$network, device = device)
  model$loss_fn = torch_load(model$loss_fn, device = device)
  model$optimizer = torch_load(model$optimizer, device = device)
  return(model)
}

#' @export
marshal_model.LearnerTorch = function(model, inplace = FALSE, ...) {
  model$model = marshal_model(model$model, inplace = inplace, ...)
  model
}

#' @export
unmarshal_model.LearnerTorch = function(model, inplace = FALSE, ...) {
  model$model = unmarshal_model(model$model, inplace = inplace, ...)
  model
}


#' @keywords internal
#' @export
hash_input.nn_module = function(x) {
  data.table::address(x)
}
