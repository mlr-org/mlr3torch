LearnerTorchClassifAbstract = R6Class("LearnerTorchClassifAbstract",
  inherit = LearnerClassif,
  public = list(
    initialize = function(id = "classif.torch.abstract", param_vals = list(),
      param_set, .optimizer, .criterion
    ) {
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  )
)
