#' @title LearnerTorchClassif
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassifTorchBase,
  public = list(
    initialize = function(id = "classif.torch", param_vals = list()) {
      super$initialize(
        id = id,
        properties = c("twoclass", "multiclass", "hotstart_forward", "weights"),
        label = "Neural Network Classifier",
        feature_types = c("logical", "integer", "numeric", "factor")
      )
    }
  )
)

mlr_learners$add("classif.torch", LearnerClassifTorch)


if (FALSE) {
  l = lrn("classif.torch", optimizer = "adam", criterion = "bce")
}
