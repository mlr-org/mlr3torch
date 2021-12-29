PipeOpClassifModel = R6::R6Class("PipeOpClassifModel",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    initialize = function(id = "classif.model", optimizer, lr) {
      self$id = id
      self$optimizer = optimizer
    }
  )
)

