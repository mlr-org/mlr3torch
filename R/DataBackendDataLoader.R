#' This is the DataBackend for a torch::dataloader
DataBackendDataLoader = R6Class("DataBackendDataLoader",
  inherit = DataBackend,
  public = list(
    initialize = function(dataloader) {
      stop("Not implemented yet!")

    }
  )
)
