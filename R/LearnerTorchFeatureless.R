#' @title Torch Featureless Learner
#'
#' @usage NULL
#' @name mlr_learners_classif.torch_featureless
#' @format `r roxy_format(LearnerClassifTorchFeatureless)`
#'
#' @description
#' Featureless classification torch learner.
#'
#' @section Construction: `r roxy_construction(LearnerClassifTorchFeatureless)`
#' @section State: See [`LearnerClassifTorch`].
#' @section Parameters: Only those from [`LearnerClassifTorch`].
#' @section Fields: `r roxy_fields_inherit(LearnerClassifTorchFeatureless)`
#' @section Methods: `r roxy_methods_inherit(LearnerClassifTorchFeatureless)`
#' @section Internals:
#' Output is a constant weight that is learner during training.
#' For classification this should result in a majority class prediction with the standard cross-entropy loss.
#' For regression, it should result in the median, for L1-loss and in the mean for L2-loss.
#'
#' @family Learners
#' @export
LearnerClassifTorchFeatureless = R6Class("LearnerClassifTorchFeatureless",
  inherit = LearnerClassifTorch,
  public = list(
    initialize = function(optimizer = t_opt("adam"), loss = t_loss("cross_entropy"), callbacks = list()) {
      param_set = ps()
      super$initialize(
        id = "classif.torch_featureless",
        label = "Torch Classification Debug Learner",
        param_set = param_set,
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
      dataset_num_categ(self, task, param_vals)
    }
  )
)

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
  forward = function(input_num = NULL, input_categ = NULL) {
    n = max(nrow(input_num), nrow(input_categ))
    self$weights$expand(c(n, -1L))
  }
)

register_learner("classif.torch_featureless", LearnerClassifTorchFeatureless)
