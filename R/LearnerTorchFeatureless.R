#' @title Featureless Torch Classifier
#'
#' @templateVar id classif.torch_featureless
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' Featureless torch learner.
#' Output is a constant weight that is learned during training.
#' For classification, this should (asymptoptically) result in a majority class prediction with the standard cross-entropy loss.
#' For regression, this should result in the median for L1 loss and in the mean for L2 loss.
#'
#' @section Parameters:
#' Only those from [`LearnerClassifTorch`].
#'
#' @export
LearnerClassifTorchFeatureless = R6Class("LearnerClassifTorchFeatureless",
  inherit = LearnerClassifTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(optimizer = t_opt("adam"), loss = t_loss("cross_entropy"), callbacks = list()) {
      super$initialize(
        id = "classif.torch_featureless",
        label = "Featureless Torch Classifier",
        param_set = ps(),
        feature_types = unname(mlr_reflections$task_feature_types),
        properties = c("twoclass", "multiclass", "missings", "featureless"),
        predict_types = c("response", "prob"),
        man = "mlr3torch::mlr_learners_classif.torch_featureless",
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      nn_featureless(nout = length(task$class_names))
    },
    .dataset = function(task, param_vals) {
      dataset_featureless(task, param_vals$device, target_batchgetter("classif"))
    }
  )
)

#' @title Featureless Torch Regression Learner
#'
#' @templateVar id regr.torch_featureless
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @inherit mlr_learners_classif.torch_featureless description
#'
#' @section Parameters:
#' Only those from [`LearnerRegrTorch`].
#'
#' @export
LearnerRegrTorchFeatureless = R6Class("LearnerRegrTorchFeatureless",
  inherit = LearnerRegrTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(optimizer = t_opt("adam"), loss = t_loss("mse"), callbacks = list()) {
      super$initialize(
        id = "classif.torch_featureless",
        label = "Featureless Torch Classifier",
        param_set = ps(),
        feature_types = unname(mlr_reflections$task_feature_types),
        properties = c("missings", "featureless"),
        predict_types = "response",
        man = "mlr3torch::mlr_learners_regr.torch_featureless",
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      nn_featureless(nout = 1L)
    },
    .dataset = function(task, param_vals) {
      dataset_featureless(task, param_vals$device, target_batchgetter("regr"))
    }
  )
)

dataset_featureless = dataset(
  initialize = function(task, device, target_batchgetter) {
    self$device = device
    self$task = task
    self$target_batchgetter = target_batchgetter
  },
  .getbatch = function(index) {
    target = self$task$data(rows = self$task$row_ids[index], cols = self$task$target_names)
    y = self$target_batchgetter(target, self$device)
    list(
      x = list(n = torch_tensor(nrow(target), dtype = torch_long(), device = self$device)),
      y = y,
      .index = index
    )

  },
  .length = function() {
    self$task$nrow
  }
)


nn_featureless = nn_module(
  initialize = function(nout) {
    self$weights = nn_parameter(torch_randn(nout))
  },
  forward = function(n) {
    # The return from the dataloader is a torch_long tensor of shape 1.
    # Apparently the dataloader does some conversions as we put in an intger
    self$weights$expand(c(n$item(), -1L))
  }
)

register_learner("classif.torch_featureless", LearnerClassifTorchFeatureless)
register_learner("regr.torch_featureless", LearnerRegrTorchFeatureless)
