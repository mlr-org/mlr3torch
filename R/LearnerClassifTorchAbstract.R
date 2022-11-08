#' @title Abstract Base Class for Torch Classification Network
#'
#' @name mlr_learners_classif.torch_abstract
#' @format [`R6Class`] object inheriting from [`LearnerClassif`].
#'
#' @description
#' This base class provides the basic functionality for training and prediction of a neural network.
#' All Torch Classification Learners should inherit from this base class.
#'
#' @section Construction:
#' `r roxy_construction(LearnerClassifTorchAbstract)`
#'
#' * `id` :: (`character(1)`)\cr
#'   The id of the learner.
#' * `optimizer` :: ([`TorchOptimizer`])\cr
#'   The optimizer for the model.
#' * `loss` :: ([`TorchLoss`])\cr
#'   The loss for the model.
#' * `param_set` (`paradox::ParamSet`)\cr
#'   Additional parameters to the standard paramset.
#' * `label` :: (`character(1)`)\cr
#'   The label for the learner.
#' * `properties` :: (`character()`)\cr
#'   The properties for the learner, see `mlr_reflections$learner_properties`.
#' * `packages` :: (`character()`)\cr
#'   The additional packages on which the learner depends.
#' * `predict_types` :: (`character()`)\cr
#'   The learner's predict types, see `mlr_reflections$learner_predict_types`.
#' * `feature_types` :: (`character()`)\cr
#'   The feature types the learner can deal with, see `mlr_reflections$task_feature_types`.
#' * `man` :: (`character(1)`)\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#'   The referenced help package can be opened via method `$help()`.
#'
#' @template paramset_torchlearner
#'
#' @section Inheriting:
#' When inheriting from this class, one should overload the `private$.network(task)` method which should construct
#' a `nn_module()` object for the given task and parameter values.
#'
#' @section Fields:
#' Fields inherited from [`LearnerClassif`] and
#' * `network` :: ([`nn_module()`][torch::nn_module])\cr
#'   The network (only available after training).
#'
#' @section Methods:
#' Only methods inherited from LearnerClassif.
#'
#' @family Learners
#' @export
LearnerClassifTorchAbstract = R6Class("LearnerClassifTorchAbstract",
  inherit = LearnerClassif,
  public = list(
    initialize = function(id, optimizer, loss, param_set, label = NULL, properties = NULL,
      packages = character(0), predict_types = NULL, feature_types, man) {
      private$.optimizer = as_torch_optimizer(optimizer, clone = TRUE)
      private$.loss = as_torch_loss(loss, clone = TRUE)
      assert_subset("classif", private$.loss$task_types)
      properties = properties %??% c("weights", "multiclass", "twoclass", "hotstart_forward")
      predict_types = predict_types %??% "response"
      label = label %??% "Classification Neural Network"

      packages = assert_character(packages, any.missing = FALSE, min.chars = 1L)
      packages = union(c("mlr3torch", "torch"), packages)


      private$.optimizer$param_set$set_id = "opt"
      private$.loss$param_set$set_id = "loss"

      p = paramset_torchlearner()

      p$values = list(
        num_threads = 1L,
        drop_last = FALSE,
        shuffle = TRUE
      )

      # TODO: the following breaks when learner is cloned; see e.g. PipeOp / PipeOpLearnerCV on how to properly handle this.
      param_set_complete = ParamSetCollection$new(list(
        param_set,
        p,
        private$.optimizer$param_set,
        private$.loss$param_set
      ))

      param_set_complete = paramset_torchlearner()
      param_set_complete$add(param_set)

      super$initialize(
        id = id,
        packages = packages,
        param_set = param_set_complete,
        predict_types = predict_types,
        properties = properties,
        data_formats = "data.table",
        label = label,
        feature_types = feature_types,
        man = man
      )
    }
  ),
  private = list(
    .train = function(task) {
      learner_torch_train(self, private, super, task)
    },
    .predict = function(task) {
      # When keep_last_prediction = TRUE we store the predictions of the last validation and we
      # therefore don't have to recompute them in the resample(), but can simple return the
      # cached predictions
      learner_torch_predict(self, private, super, task)
    },
    .network = function(task) stop(".network must be implemented."),
    # the dataloader gets param_vals that may be different from self$param_set$values, e.g.
    # when the dataloader for validation data is loaded, `shuffle` is set to FALSE.
    .dataloader = function(task, param_vals) stop(".dataloader must be implemented."),
    .optimizer = NULL,
    .loss = NULL
  ),
  active = list(
    network = function(rhs) {
      assert_ro_binding(rhs)
      if (is.null(self$state)) {
        stopf("Cannot access network before training.")
      }
      self$state$network
    }
  )
)

paramset_torchlearner = function() {
  ps(
    batch_size            = p_int(tags = c("train", "predict"), lower = 1L, default = 1L),
    epochs                = p_int(tags = c("train", "hotstart", "required"), lower = 0L),
    device                = p_fct(tags = c("train", "predict"), levels = c("auto", "cpu", "cuda", "meta"), default = "auto"), # nolint
    measures_train        = p_uty(tags = "train", custom_check = check_measures),
    measures_valid        = p_uty(tags = "train", custom_check = check_measures),
    augmentation          = p_uty(tags = "train"),
    callbacks             = p_uty(tags = "train", custom_check = check_callbacks),
    drop_last             = p_lgl(default = FALSE, tags = "train"),
    keep_last_prediction  = p_lgl(default = TRUE, tags = "train"),
    num_threads           = p_int(default = 1L, lower = 1L, tags = c("train", "predict", "threads")),
    shuffle               = p_lgl(default = TRUE, tags = "train"),
    early_stopping_rounds = p_int(default = 0L, tags = "train")
  )
}
