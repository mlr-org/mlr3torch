LearnerClassifAlexNet = R6Class("LearnerClassifAlexNet",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function() {
      param_set = ps(
        pretrained = p_lgl(default = FALSE, tags = "train")

      )
    }
  )
)
