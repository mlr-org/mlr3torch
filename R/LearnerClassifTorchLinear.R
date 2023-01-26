#' @title Linear Torch Classifier
#'
#' @usage NULL
#' @format `r roxy_format(LearnerClassifTorchLinear)`
#' @name mlr_learners_classif.alexnet
#'
#' @description
#' Linear torch classification network, mostly for testing purposes.
#'
#' @section Parameters:
#' Only parameters inherited from [`LearnerClassifTorchAbstract`].
#'
#' @export
LearnerClassifTorchLinear = R6Class("LearnerClassifTorchLinear",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function(optimizer = t_opt("adam"), loss = t_loss("cross_entropy")) {
      super$initialize(
        id = "classif.torch_linear",
        optimizer = optimizer,
        loss = loss,
        param_set = ps(),
        feature_types = c("integer", "numeric"),
        man = "mlr3learners.torch::mlr_learners_torch_linear",
        label = "Linear Torch Classifier"
      )

    }
  ),
  private = list(
    .network = function(task, param_vals) {
      nn_linear(length(task$feature_names), length(task$class_names))
    },
    .dataset = function(task, param_vals) {
      ingress_token = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, length(task$feature_names)))
      dataset = task_dataset(
        task,
        feature_ingress_tokens = list(num = ingress_token),
        target_batchgetter = crate(function(data, device) {
          torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
        }, .parent = topenv()),
        device = param_vals$device
      )
    }
  )
)

register_learner("classif.torch_linear", LearnerClassifTorchLinear)
