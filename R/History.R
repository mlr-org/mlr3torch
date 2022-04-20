History = R6Class("History",
  public = list(
    train_loss = NULL,
    test_loss = NULL,
    initialize = function(epochs, train_ids, valid_ids, drop_last) {
      self$train_loss = list()
      self$test_loss = list()
    }
  )
)
