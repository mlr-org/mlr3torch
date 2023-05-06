#' @title Featureless Torch Classifier
#'
#' @templateVar id classif.torch_featureless
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' Featureless torch learner.
#' Output is a constant weight that is learner during training.
#' For classification this should result in a majority class prediction with the standard cross-entropy loss.
#' For regression, it should result in the median, for L1 loss and in the mean for L2 loss.
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
        # TODO: This should have all feature types, and have properties missing
        feature_types = c("integer", "numeric", "factor", "ordered"),
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
      nn_featureless(task)
    },
    .dataset = function(task, param_vals) {
      # TODO: Here we don't need the X values
      # also adjust the feature types then
      # the dataloader should just return the number of observations for the output shape and the target
      dataset_num_categ(self, task, param_vals)
    }
  )
)

# TODO: Regression

nn_featureless = nn_module(
  initialize = function(task) {
    if (task$task_type == "classif") {
      n_out = length(task$class_names)
      self$weights = nn_parameter(torch_randn(n_out))
    } else if (task$task_type == "regr") {
      self$weights = nn_parameter(torch_randn(1))
    } else {
      stopf("Unsupported task type.")
    }
  },
  # TODO: This should take in only an integer n
  forward = function(input_num = NULL, input_categ = NULL) {
    n = max(nrow(input_num), nrow(input_categ))
    self$weights$expand(c(n, -1L))
  }
)

register_learner("classif.torch_featureless", LearnerClassifTorchFeatureless)
