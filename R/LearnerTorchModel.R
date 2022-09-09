
#' @title LearnerClassifTorchModel
#' @export
LearnerClassifTorchModel = R6Class("LearnerClassifTorchModel",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(network, ingress_tokens, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"), feature_types = NULL, packages = character(0)) {
      private$.network_stored = assert_class(network, "nn_module")
      private$.ingress_tokens = assert_list(ingress_tokens, types = "TorchIngressToken")
      super$initialize(
        id = "classif.torch",
        properties = c("weights", "twoclass", "multiclass", "hotstart_forward"),
        label = "Torch Classification Network",
        optimizer = optimizer,
        loss = loss,
        packages = packages,
        predict_types = c("response", "prob"),
        feature_types = feature_types %??% mlr_reflections$task_feature_types,
        man = "mlr3torch::mlr_learners_classif.torch"
      )
    }
  ),
  private = list(
    .network = function(task) {
      private$.network_stored
    },
    .dataloader = function(task, param_vals) {
      dataset = task_dataset(
        task,
        feature_ingress_tokens = private$.ingress_tokens,
        target_batchgetter = crate(function(data, device) {
          torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
        }, .parent = topenv()),
        device = param_vals$device %??% "auto"
      )
      dataloader(
        dataset = dataset,
        batch_size = param_vals$batch_size %??% 1,
        drop_last = param_vals$drop_last %??% FALSE,
        shuffle = param_vals$shuffle %??% TRUE
      )
    },
    .network_stored = NULL,
    .ingress_tokens = NULL
  )
)

