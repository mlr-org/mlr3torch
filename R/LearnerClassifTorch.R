#' @title LearnerTorchClassif
#' @export
LearnerClassifTorch = R6Class("LearnerClassifTorch",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    #' @param .optimizer (`character(1)`)\cr
    #'   The optimizer, see `torch_reflections$optimizer`.
    #' @param .loss (`character(1)`)\cr
    #'   The loss, see `torch_reflections$loss`.
    initialize = function(id = "classif.torch", param_vals = list(), .optimizer, .loss) {
      super$initialize(
        id = id,
        properties = c("twoclass", "multiclass", "hotstart_forward", "weights"),
        label = "Neural Network Classifier",
        feature_types = c("logical", "integer", "numeric", "factor"),
        optimizer = .optimizer,
        loss = .loss,
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
    }
  )
)
