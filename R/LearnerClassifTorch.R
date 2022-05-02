#' @title LearnerTorchClassif
#' @export
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function(id = "classif.torch", param_vals = list(), .optimizer, .loss) {
      super$initialize(
        id = id,
        properties = c("twoclass", "multiclass", "hotstart_forward", "weights"),
        label = "Neural Network Classifier",
        feature_types = c("logical", "integer", "numeric", "factor"),
        .optimizer = .optimizer,
        .loss = .loss,
        man = "mlr3torch::mlr_learners_classif.torch"
      )
    }
  ),
  private = list(
    .network = function(task) {
      architecture = self$param_set$values$architecture

      if (test_r6(architecture, "Architecture")) {
        network = architecture$build(task)
      } else if (test_r6(architecture, "nn_Module")) {
        network = architecture$clone(deep = TRUE)
      } else {
        stopf("Invalid argument for architecture.")
      }
    },
    .optimizer = NULL
  )
)
