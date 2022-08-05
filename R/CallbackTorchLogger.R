CallbackTorchLogger = R6Class("CallbackTorchLogger",
  inherit = CallbackTorch,
  public = list(
    measures_train = NULL,
    measures_valid = NULL,
    initialize = function(measures_train, measures_valid) {
      self$measures_train = measures_train
      self$measures_valid = measures_valid
    },
    on_start = function(self, context) {
      if (get_private(context)$.epoch) {
        get_private(context)$.epoch = 1L
      } else {
        get_private(context)$.epoch = context$epoch + 1L
      }
    },
    on_before_train_epoch = function(self, context) {
      get_private(context)$.train_iter = 1L
      context$history$train_loss[[context$epoch]] = numeric(length(context$train_loader))
    },
    on_after_train_batch = function(self, context) {
      context$history$train_loss[[context$epoch]][context$train_iter] = context$y_hat
      get_private(context)$.train_iter = context$train_iter + 1L
    },
    on_after_train_epoch = function(self, context) {
      get_private(context)$.train_iter = NULL
    },
    on_before_valid_epoch = function(self, context) {
      get_private(context)$.valid_iter = 1L
    },
    on_after_valid_batch = function(self, context) {
      get_private(context)$.valid_iter = context$valid_iter + 1L
    },
    on_after_valid_epoch = function(self, context) {
      get_private(context)$.valid_iter = NULL
    }
  )

)
