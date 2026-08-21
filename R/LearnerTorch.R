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
#' * `"predefined"`: This will use the predefined `$internal_valid_task` of a [`mlr3::Task`].
#'
#' This validation data can also be used for early stopping, see the description of the `Learner`'s parameters.
#'
#' @section Saving a Learner:
#' In order to save a `LearnerTorch` for later usage, it is necessary to call the `$marshal()` method on the `Learner`
#' before writing it to disk, as the object will otherwise not be saved correctly.
#' After loading a marshaled `LearnerTorch` into R again, you then need to call `$unmarshal()` to transform it
#' into a useable state.
#'
#' @section Early Stopping and Internal Tuning:
#' In order to prevent overfitting, the `LearnerTorch` class allows to use early stopping via the `patience`
#' and `min_delta` parameters, see the `Learner`'s parameters.
#' When tuning a `LearnerTorch` it is also possible to combine the explicit tuning via `mlr3tuning`
#' and the `LearnerTorch`'s internal tuning of the epochs via early stopping.
#' To do so, you just need to include `epochs = to_tune(upper = <upper>, internal = TRUE)` in the search space,
#' where `<upper>` is the maximally allowed number of epochs, and configure the early stopping.
#'
#' @section Checkpointing and Resuming:
#' It is possible to save intermediate results from a run via the 
#' [`t_clbk("checkpoint")`][mlr_callback_set.checkpoint] callback.
#' It is then possible to train for more epochs by setting the `resume` parameter of the `LearnerTorch`.
#' This parameter can either be a path or `TRUE` which will use the path of the provided checkpoint
#' callback.
#' Only the number of epochs should be changed between resumed runs, other parameter changes
#' are considered undefined behavior.
#' Also, make sure to use the same train-validation split.
#' When the latest written checkpoint was for `n1` epochs, the learner needs to be configured
#' to be trained for `n >= n1` epochs and the training will run for `n2 = n - n1` epochs.
#' With `n = n1` the checkpointed run is already finished, so nothing is trained and the model of
#' the checkpoint is returned -- which is what lets a script that restarts itself be run again
#' after it succeeded, and what recovers the model of a run that was killed after its last epoch.
#' Configuring `n < n1` is an error.
#' Resuming will load the network weights, optimizer states and callback states.
#' For some callbacks, training for `n1` and then `n2` epochs via resuming is not the same as
#' training for `n` epochs from the start.
#' This is for example the case for learning rate schedulers that depend on the total number of epochs
#' to train for.
#' The callbacks document their behavior under a corresponding
#' *Resuming* section in their documentation.
#' Furthermore, rng states are not restored, which constitutes another difference between a full
#' and a resumed training run.
#'
#'
#' @section Network Head and Target Encoding:
#' Torch learners are expected to have the following output:
#' * binary classification: `(batch_size, 1)`, representing the logits for the positive class.
#' * multiclass classification: `(batch_size, n_classes)`, representing the logits for all classes.
#' * regression: `(batch_size, 1)` representing the response prediction.
#'
#' A network may return more than one tensor, in which case it returns a `list()` of them.
#' There are two typical reasons for this:
#' * Networks with auxiliary classifiers such as [`Inception v3`][mlr_learners.torchvision] return
#'   additional predictions that only exist to contribute to the loss during training. Here, every
#'   element has the shape given above and the first one is the prediction of interest.
#' * A prediction that consists of several quantities -- e.g. a mean and a standard deviation -- is
#'   expressed by returning one tensor per quantity, which the prediction encoding combines.
#'
#' In both cases the complete output is what the rest of the learner works with:
#' * The loss is applied to it. Because the configured loss expects a single tensor, a learner
#'   whose network returns a list has to wrap it by overloading `.loss_fn()`, see the list of
#'   methods below. [`ContextTorch`] makes the output available as `ctx$y_hats`.
#' * The prediction is encoded from it, both when predicting and when calculating the training and
#'   validation scores, so `.encode_prediction()` always receives the complete network output. Such
#'   a learner needs an [`encode_prediction()`] method for its task type, or has to overload the
#'   private `.encode_prediction()` method, because the encodings of the built-in task types expect
#'   a single tensor.
#'
#' Note that the complete output is whatever the network returned in the mode it was called in, so a
#' network whose extra tensors exist only during training -- as auxiliary classifiers do -- returns a
#' different structure during training than during prediction, and `.encode_prediction()` has to
#' handle both. `classif.inception_v3` does this by encoding only the prediction of the main
#' classifier.
#'
#' Furthermore, the target encoding is expected to be as follows:
#' * regression: The `numeric` target variable of a [`TaskRegr`][mlr3::TaskRegr] is encoded as a
#'   [`torch_float`][torch::torch_float] with shape `c(batch_size, 1)`.
#' * binary classification: The `factor` target variable of a [`TaskClassif`][mlr3::TaskClassif] is encoded as a
#'   [`torch_float`][torch::torch_float] with shape `(batch_size, 1)` where the positive class (`Task$positive`, which
#'   is also ensured to be the first factor level) is `1` and the negative class is `0`.
#' * multi-class classification: The `factor` target variable of a [`TaskClassif`][mlr3::TaskClassif] is a label-encoded
#'   [`torch_long`][torch::torch_long] with shape `(batch_size)` where the label-encoding goes from `1` to `n_classes`.
#'
#' @section Predicting Tensors:
#' The predict type `"lazy_tensor"`, available for the task type `"torch"`, hands back what the
#' network produced -- a [`lazy_tensor`] with one element per observation -- instead of asking the
#' task's `default_encoder` to turn it into a response.
#' It is how to get at the logits of a classifier or the reconstruction of an autoencoder, and a
#' task predicted this way needs no encoder at all.
#' Like every predict type of this task type it is opt-in:
#' `lrn("torch.module", predict_types = c("response", "lazy_tensor"))`.
#' A network with more than one head hands back one [`lazy_tensor`] per head, held in a
#' [`data.table`][data.table::data.table] with one column per head so that the prediction is still
#' one row per observation; `as.data.table()` spreads it into `lazy_tensor.<head>` columns.
#'
#' Two things to know before using it:
#' * **Such a prediction does not survive [`saveRDS()`][base::saveRDS].** It holds `torch` tensors,
#'   which are external pointers: saving *succeeds*, and the object then fails with
#'   *external pointer is not valid* the next time the tensors are touched, in this session or in
#'   another. This applies to a [`ResampleResult`][mlr3::ResampleResult] holding one as well -- its
#'   row ids and scores survive, its tensors do not.
#' * Nothing about it is lazy. A [`lazy_tensor`] built from a tensor holds that tensor, so a
#'   prediction of this type is the network's output in memory, and `resample()` holds every fold's
#'   -- combining the folds concatenates them, since lazy tensors from different networks share no
#'   data descriptor and cannot be concatenated lazily.
#'
#' @section Important Runtime Considerations:
#' There are a few hyperparameters settings that can have a considerable impact on the runtime of the learner.
#' These include:
#'
#' * `device`: Use a GPU if possible.
#' * `num_threads`: Set this to the number of CPU cores available if training on CPU.
#'   When resampling, benchmarking or tuning in parallel, each worker uses `num_threads` threads, so
#'   divide the available cores among the workers instead to avoid oversubscribing the machine.
#' * `tensor_dataset`: Set this to `TRUE` (or `"device"` if on a GPU) if the dataset fits into memory.
#' * `batch_size`: Especially for very small models, choose a larger batch size.
#'
#' Also, see the *Early Stopping and Internal Tuning* section for how to terminate training early.
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
#'   For the task type `"torch"`, it defaults to `"response"`.
#'   For other task types, it defaults to all predict types that are registered for the task type.
#'   To deviate from the defaults, it is necessary to overwrite the private `$.encode_prediction()`
#'   method, see section *Inheriting*.
#' @param loss (`NULL` or [`TorchLoss`])\cr
#'   The loss to use for training.
#'   Defaults to MSE for regression and cross entropy for classification.
#'   For other task types there is no default and the loss has to be given, because which loss is
#'   appropriate depends on the learning problem.
#' @param optimizer (`NULL` or [`TorchOptimizer`])\cr
#'   The optimizer to use for training.
#'   Defaults to adam.
#' @param callbacks (`list()` of [`TorchCallback`]s)\cr
#'   The callbacks to use for training.
#'   Defaults to an empty` list()`, i.e. no callbacks.
#'   Within a stage they are called in the order in which they are provided, unless a callback
#'   requests otherwise via its `$weight`, see section *Ordering* of [`CallbackSet`].
#' @param jittable (`logical(1)`)\cr
#'   Whether the model can be jit-traced. Default is `FALSE`.
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
#' There are no separate classes for classification and regression to inherit from.
#' Instead, the `task_type` must be specified as a construction argument.
#' Any task type that is registered in
#' [`mlr_reflections$task_types`][mlr3::mlr_reflections] can be used.
#' Support for a task type that \CRANpkg{mlr3torch} does not know is added by implementing methods for
#' the three S3 generics that hold the task-type-specific behaviour: [`output_dim_for()`] (how many
#' output neurons the network needs), [`get_target_batchgetter()`] (how the target is turned into a
#' tensor) and [`encode_prediction()`] (how the network's output is turned back into a prediction).
#' Such a learner also has to be given a `loss` explicitly.
#' This class can also be used for custom task types, see [`TaskTorch`] and the
#' *Custom Learning Problems* article for more information.
#'
#' When inheriting from this class, one should overload the following methods:
#'
#' * `.network(task, param_vals)`\cr
#'   ([`Task`][mlr3::Task], `list()`) -> [`nn_module`][torch::nn_module]\cr
#'   Construct a [`torch::nn_module`] object for the given task and parameter values, i.e. the neural network that
#'   is trained by the learner.
#'   Note that a specific output shape is expected from the returned network, see section *Network Head and Target Encoding*.
#'   That section also describes when a network can return more than one tensor.
#'   You can use [`output_dim_for()`] to obtain the correct output dimension for a given task.
#' * `.loss_fn(task, param_vals)`\cr
#'   ([`Task`][mlr3::Task], `list()`) -> [`nn_module`][torch::nn_module]\cr
#'   Construct the loss that is applied to the output of the network.
#'   The default implementation generates the loss that was configured by the user, i.e.
#'   `self$loss$generate(task)`.
#'   Overload this if the network returns more than one prediction and the configured loss has to
#'   be wrapped, see the `aux_logits` parameter of
#'   [`classif.inception_v3`][mlr_learners.torchvision].
#' * `.ingress_tokens(task, param_vals)`\cr
#'   ([`Task`][mlr3::Task], `list()`) -> named `list()` with [`TorchIngressToken`]s\cr
#'   Create the [`TorchIngressToken`]s that are passed to the [`task_dataset`] constructor.
#'   The number of ingress tokens must correspond to the number of input parameters of the network.
#'   If there is more than one input, the names must correspond to the inputs of the network.
#'   See [`ingress_num`], [`ingress_categ`], and [`ingress_ltnsr`] on how to easily create the correct tokens.
#'   For more flexibility, you can also directly implement the `.dataset(task, param_vals)` method,
#'   see below.
#' * `.dataset(task, param_vals)`\cr
#'   ([`Task`][mlr3::Task], `list()`) -> [`torch::dataset`]\cr
#'   Create the dataset for the task.
#'   Don't implement this if the `.ingress_tokens()` method is defined.
#'   The dataset must return a named list where:
#'   * `x` is a list of torch tensors that are the input to the network.
#'     For networks with more than one input, the names must correspond to the inputs of the network.
#'   * `y` is the target tensor.
#'   * `.index` are the indices of the batch (`integer()` or a `torch_int()`).
#'
#'   For information on the expected target encoding of `y`, see section *Network Head and Target Encoding*.
#'   Moreover, one needs to pay attention respect the row ids of the provided task.
#'   It is recommended to relu on [`task_dataset`] for creating the [`dataset`][torch::dataset].
#'
#' It is also possible to overwrite the private `.dataloader()` method.
#' This must respect the dataloader parameters from the [`ParamSet`][paradox::ParamSet].
#'
#' * `.dataloader(dataset, param_vals)`\cr
#'   ([`dataset`][torch::dataset], `list()`) -> [`torch::dataloader`]\cr
#'   Create a dataloader from the dataset.
#'   Needs to respect at least `batch_size` and `shuffle` (otherwise predictions will be incorrectly ordered).
#'   Use `get_batch_size(param_vals, "train")` to obtain the batch size for the respective phase,
#'   which takes the `batch_size_predict` parameter into account.
#'
#' To change the predict types, it is possible to overwrite the method below:
#'
#' * `.encode_prediction(network_output, task)`\cr
#'   ([`torch_tensor`][torch::torch_tensor] or `list()` of them, [`Task`][mlr3::Task]) -> `list()`\cr
#'   Take in the raw predictions from `self$network` (`network_output`) and encode them into a
#'   format that can be converted to valid `mlr3` predictions using [`mlr3::as_prediction_data()`].
#'   It is a `list()` of tensors when the network returns more than one, see section
#'   *Network Head and Target Encoding*.
#'   This method must take `self$predict_type` into account.
#'
#' While it is possible to add parameters by specifying the `param_set` construction argument, it is currently
#' not possible to remove existing parameters, i.e. those listed in section *Parameters*.
#' None of the parameters provided in `param_set` can have an id that starts with `"loss."`, `"opt."`,
#' or `"cb."`, as these are preserved for the dynamically constructed parameters of the optimizer, the loss function,
#' and the callbacks.
#'
#' To perform additional input checks on the task, the private `.check_train_task(task, param_vals)` and
#' `.check_predict_task(task, param_vals)` can be overwritten.
#' These should return `TRUE` if the input task is valid and otherwise a string with an error message.
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
    initialize = function(id, task_type, param_set, properties = character(), man, label, feature_types,
      optimizer = NULL, loss = NULL, packages = character(), predict_types = NULL, callbacks = list(),
      jittable = FALSE) {
      assert_choice(task_type, mlr_reflections$task_types$type)

      predict_types = predict_types %??% switch(task_type,
        regr = "response",
        classif = c("response", "prob"),
        torch = "response",
        names(mlr_reflections$learner_predict_types[[task_type]])
      )

      assert_subset(properties, mlr_reflections$learner_properties[[task_type]])
      properties = union(properties, c("marshal", "validation", "internal_tuning"))
      if (task_type == "classif") {
        properties = union(properties, c("twoclass", "multiclass"))
      }
      assert_subset(predict_types, names(mlr_reflections$learner_predict_types[[task_type]]))
      packages = assert_character(packages, any.missing = FALSE, min.chars = 1L)
      packages = union(c("mlr3", "mlr3torch"), packages)

      private$.param_set_torch = paramset_torchlearner(task_type, jittable = jittable)

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
          assert_true(grepl("^(self|private|super)", deparse(x)),
            .var.name = "Don't use self, private, or super in param_set")
          check_ps(eval(x))
        })
        private$.param_set_source = param_set
      }
      # loss needs access to the task_type
      self$task_type = task_type
      if (is.null(loss)) {
        default_loss = switch(task_type, classif = "cross_entropy", regr = "mse", NULL)
        if (is.null(default_loss)) {
          stopf("There is no default loss for task type '%s', pass the `loss` explicitly.", task_type) # nolint
        }
        private$.loss = t_loss(default_loss)
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

      if ("early_stopping" %in% ids(self$callbacks)) {
        stopf("Callback with id 'early_stopping' is reserved.")
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
        label = label,
        feature_types = feature_types,
        man = man
      )

    },
    #' @description
    #' Helper for print outputs.
    #' @param ... (ignored).
    format = function(...) {
      sprintf("<%s:%s>", class(self)[1L], self$id)
    },

    #' @description
    #' Prints the object.
    #' @param ... (any)\cr
    #'   Currently unused.
    print = function(...) {
      super$print(...)
      mlr3misc::cat_cli({
        cli::cli_li("Optimizer: {private$.optimizer$id}")
        cli::cli_li("Loss: {private$.loss$id}")
        cli::cli_li(paste0("Callbacks: ", if (length(private$.callbacks)) as_short_string(paste0(ids(private$.callbacks), collapse = ","), 1000L) else "-"))
      })
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
    },
    #' @description
    #' The raw output of the trained network on a `task`, i.e. what the private
    #' `$.encode_prediction()` method is handed before it turns it into a
    #' [`Prediction`][mlr3::Prediction]: the logits of a classifier, the reconstruction of an
    #' autoencoder, or whatever else the network returns.
    #' This runs the same path as `$predict()` -- evaluation mode, device placement, batching and
    #' [`with_no_grad()`][torch::with_no_grad] -- so it is not the same as calling `$network`
    #' yourself, which leaves all four to you.
    #' @param task ([`Task`][mlr3::Task])\cr
    #'   The task to predict on.
    #' @param row_ids (`integer()` or `NULL`)\cr
    #'   The rows to predict on. All rows if `NULL` (default).
    #' @return [`torch_tensor`][torch::torch_tensor], or a `list()` of them for a network that
    #'   returns more than one, see section *Network Head and Target Encoding*.
    predict_tensor = function(task, row_ids = NULL) {
      assert_task(task)
      if (is.null(self$model)) {
        stopf("Learner '%s' has not been trained yet, so it has no network to predict with.", self$id)
      }
      if (!is.null(row_ids)) {
        task = task$clone(deep = TRUE)$filter(assert_row_ids(row_ids))
      }
      param_vals = self$param_set$get_values(tags = "predict")
      param_vals$device = auto_device(param_vals$device)
      with_torch_settings(seed = self$model$seed, num_threads = param_vals$num_threads,
        num_interop_threads = param_vals$num_interop_threads, expr = {
        learner_torch_network_output(self, private, task, param_vals)
      })
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
        if (self$task_type != "torch") {
          assert_choice(self$task_type, loss$task_types)
        }
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
        self$packages = unique(c(self$packages, unlist(map(private$.callbacks, "packages")))) %??% character(0)
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
    #' When early stopping is active, this returns a named list with the early-stopped epochs,
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
        return(named_list())
      }
      list(epochs = self$model$callbacks$early_stopping$best_epochs)
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
      # $train() compares task types but also checks inheritance and any torch learner inherits from
      # LearnerTorch and LearnerTorch is registered as lerner class for task type "torch" so we need
      # this additional check here
      if (task$task_type != self$task_type) {
        stopf("Learner '%s' is for task type '%s', but task '%s' has task type '%s'.", self$id, self$task_type, task$id, task$task_type) # nolint
      }
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
      msg = private$.check_train_task(task, param_vals)
      if (!isTRUE(msg)) {
        stopf("Training task '%s' is invalid for learner '%s': %s", task$id, self$id, msg)
      }

      param_vals$device = auto_device(param_vals$device)
      if (identical(param_vals$seed, "random")) param_vals$seed = sample.int(.Machine$integer.max, 1)

      model = with_torch_settings(seed = param_vals$seed, num_threads = param_vals$num_threads,
       num_interop_threads = param_vals$num_interop_threads, expr = {
        learner_torch_train(self, private, super, task, param_vals)
      })
      model$task_col_info = copy(task$col_info[c(task$feature_names, task$target_names), c("id", "type", "levels")])
      return(model)
    },
    .predict = function(task) {
      param_vals = self$param_set$get_values(tags = "predict")
      param_vals$device = auto_device(param_vals$device)
      msg = private$.check_predict_task(task, param_vals)
      if (!isTRUE(msg)) {
        stopf("Prediction task '%s' is invalid for learner '%s': %s", task$id, self$id, msg)
      }

      pdata = with_torch_settings(seed = self$model$seed, num_threads = param_vals$num_threads,
        num_interop_threads = param_vals$num_interop_threads, expr = {
        learner_torch_predict(self, private, super, task, param_vals)
      })
      # `mlr3` calls `as_prediction_data()` on this, and dispatches the ground truth on the class
      # of the task, which a `TaskTorch` is not the right kind of -- see `?TaskTorch`
      class(pdata) = c("prediction_torch", "list")
      pdata
    },
    .encode_prediction = function(network_output, task) {
      encode_prediction(
        task = task,
        network_output = network_output,
        predict_type = self$predict_type
      )
    },
    .network = function(task, param_vals) stop(".network must be implemented."),
    # Constructs the loss that is applied to the output of the network. Learners whose network
    # returns more than one prediction can overwrite this to wrap the loss that was configured
    # by the user, see e.g. the auxiliary classifier of `classif.inception_v3`.
    .loss_fn = function(task, param_vals) {
      self$loss$generate(task)
    },
    # the dataloader gets param_vals that may be different from self$param_set$values, e.g.
    # when the dataloader for validation data is loaded, `shuffle` is set to FALSE.
   .dataloader = function(dataset, param_vals) {
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
      args$batch_size = get_batch_size(param_vals, "train")

      if (!is.null(args$sampler) && !is.null(args$batch_sampler)) {
        error_config("Providing both a 'sampler' and a 'batch_sampler' is not supported, set only one of them.")
      }
      if (is.null(args$batch_sampler)) {
        if (is.null(args$batch_size)) {
          error_config("Parameter 'batch_size' must be set for training, unless a 'batch_sampler' is provided.")
        }
      } else {
        # the batch sampler already determines the batches, so these are ignored by torch::dataloader()
        args$batch_size = NULL
        args$shuffle = NULL
        args$drop_last = NULL
      }
      if (!is.null(args$sampler)) {
        # the sampler determines the order in which the observations are drawn
        args$shuffle = NULL
      }

      for (param_name in c("sampler", "batch_sampler")) {
        param_val = args[[param_name]]
        if (!is.null(param_val)) {
          # instantiate these params which should be classes.
          args[[param_name]] = param_val(dataset)
        }
      }
      invoke(dataloader, dataset = dataset, .args = args)
    },
    .dataloader_predict = function(dataset, param_vals) {
      batch_size = get_batch_size(param_vals, "predict")
      if (is.null(batch_size)) {
        error_config("Parameter 'batch_size' or 'batch_size_predict' must be set for prediction (this includes the validation data during training).")
      }
      param_vals_test = insert_named(param_vals,
        list(batch_size = batch_size, shuffle = FALSE, drop_last = FALSE))
      param_vals_test$batch_size_predict = NULL
      # samplers are only used during training, as they can change the order of the observations,
      # which would misalign the predictions with the rows of the task
      param_vals_test$sampler = NULL
      param_vals_test$batch_sampler = NULL
      private$.dataloader(dataset, param_vals_test)
    },
    .ingress_tokens = function(task, param_vals)  {
      stopf("Private method `$.ingress_tokens()` must be implemented.")
    },
    .dataset = function(task, param_vals) {
      if (!is.null(private$.ingress_tokens)) {
        task_dataset(
          task = task,
          feature_ingress_tokens = private$.ingress_tokens(task, param_vals),
          target_batchgetter = get_target_batchgetter(task)
        )
      } else {
        stopf("Private method `$.dataset()` or `$.ingress_tokens()` must be implemented.")
      }
    },
    .optimizer = NULL,
    .loss = NULL,
    .callbacks = NULL,
    .check_train_task = function(task, param_vals) TRUE,
    .check_predict_task = function(task, param_vals) TRUE,
    deep_clone = function(name, value) {
      private$.param_set = NULL # required to keep clone identical to original, otherwise tests get really ugly
      if (is.R6(value)) {
        return(value$clone(deep = TRUE))
      } else if (test_class(value, "nn_module_generator")) {
        value
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
          if (is_marshaled_model(model)) {
            # a marshaled model contains no external pointers, so the regular deep clone above is
            # already sufficient and the torch objects it would clone do not exist in this state
            value$model = model
            return(value)
          }
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
  model$jitted = inherits(model$network, "script_module")
  model$network = if (model$jitted) {
    jit_serialize(model$network)
  } else {
    torch_serialize(model$network)
  }
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
  model$network = if (isTRUE(model$jitted)) {
    deser = jit_unserialize(model$network)
    deser$to(device = device)
    deser
  } else {
    torch_load(model$network, device = device)
  }
  model$jitted = NULL
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
hash_input.nn_module_generator = function(x) {
  # A nn_module_generator is a function that holds an R6ClassGenerator as it's attribute.
  # Our default hash_input.function does not respect this, so we need a specialized
  # implementation for this.
  # We also can't hash the generator directly, because digest() hashes the serialized object
  # which depends on whether the generator's methods have been jit compiled, which changes
  # after a module was used.
  generator = attr(x, "module")
  methods = if (inherits(generator, "R6ClassGenerator")) generator$public_methods
  list(class(x), map(methods, hash_input))
}
