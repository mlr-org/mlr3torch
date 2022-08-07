LearnerClassifTorchModel = R6Class("LearnerClassifTorchModel",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(network, optimizer, loss, feature_types = NULL, packages = character(0)) {
      private$.network_stored = network
      super$initialize(
        id = "classif.torch",
        properties = c("weights", "twoclass", "multiclass", "hotstart_forward"),
        label = "Torch Classification Network",
        optimizer = optimizer,
        loss = loss,
        packages = packages,
        feature_types = feature_types %??% mlr_reflections$task_feature_types,
        man = "mlr3torch::mlr_learners_classif.torch"
      )
    }
  ),
  private = list(
    .network = function(task) {
      private$.network_stored
    },
    .network_stored = NULL
  )
)

