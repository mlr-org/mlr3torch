LearnerClassifTorchModel = R6Class("LearnerClassifTorchModel",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(network, optimizer = "adam", loss = "cross_entropy", param_vals = list(),
      feature_types = NULL) {
      private$..network = network
      super$initialize(
        id = "classif.torch",
        properties = c("weights", "twoclass", "multiclass", "hotstart_forward"),
        label = "Classification Network",
        feature_types = feature_types %??% mlr_reflections$task_feature_types,
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners_classif.torch",
        param_set = ps()
      )
    }
  ),
  private = list(
    .network = function(task) {
      private$..network
    },
    ..network = NULL
  )
)

