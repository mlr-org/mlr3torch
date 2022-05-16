#' Dictionary of TorchOps
#' @export
mlr_torchops = R6Class("DictionaryTorchOp",
  inherit = mlr3misc::Dictionary,
  cloneable = FALSE,
  public = list(
    metainf = new.env(parent = emptyenv()),
    add = function(key, value, metainf = NULL) {
      ret = super$add(key, value)
      if (!is.null(metainf)) {
        # we save the *expression*, not the value, because we could otherwise get version conflicts from objects.
        assign(x = key, value = substitute(metainf), envir = self$metainf)
      }
      invisible(self)
    },
    get = function(key, ...) {
      obj = get0(key, envir = self$items, inherits = FALSE, ifnotfound = NULL)
      if (is.null(obj)) {
        stopf("Element with key '%s' not found!%s", key, did_you_mean(key, self$keys()))
      }
      obj$value$new(...)
    }
  )
)$new()
