#' @description Neural Network Architecture
#' @export
#'
Architecture = R6Class("Architecture",
  public = list(
    layers = list(),
    initialize = function() {
    },
    append = function(layer) {
      self$layers = c(self$layers, layer)
      invisible(self)
    }
  )
)

if (FALSE) {
  architecture = Architecture$new()
  architecture

}
