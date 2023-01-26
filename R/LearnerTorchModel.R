#' @title Classification Torch Learner
#'
#' @usage NULL
#' @name mlr_learners_classif.torch_model
#' @format `r roxy_format(LearnerClassifTorchModel)`
#'
#' @description
#' Classification Torch Learner that is used internally by [`PipeOpTorchModelClassif`].
#'
#' @section Construction: `r roxy_construction(LearnerClassifTorchModel)`
#' TODO : The construction arguments
#' @section State: See [`LearnerClassifTorchAbstract`]
#' @section Parameters: Only those from [`LearnerClassifTorchAbstract`]
#' @section Fields: `r roxy_fields(LearnerClassifTorchModel)`
#' @section Methods: `r roxy_methods(LearnerClassifTorchModel)`
#'
#' @export
LearnerClassifTorchModel = R6Class("LearnerClassifTorchModel",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function(network, ingress_tokens, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
      packages = character(0)) {
      private$.network_stored = assert_class(network, "nn_module")
      private$.ingress_tokens = assert_list(ingress_tokens, types = "TorchIngressToken")
      super$initialize(
        id = "classif.torch_model",
        label = "Torch Classification Learner",
        optimizer = optimizer,
        loss = loss,
        packages = packages,
        param_set = ps(),
        feature_types = mlr_reflections$task_feature_types,
        man = "mlr3torch::mlr_learners_classif.torch_model"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      private$.network_stored
    },
    .dataset = function(task, param_vals) {
      dataset = task_dataset(
        task,
        feature_ingress_tokens = private$.ingress_tokens,
        target_batchgetter = crate(function(data, device) {
          torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
        }, .parent = topenv()),
        device = param_vals$device %??% self$param_set$default$device
      )
    },
    .network_stored = NULL,
    .ingress_tokens = NULL
  )
)
