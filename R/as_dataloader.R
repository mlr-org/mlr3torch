#' @title Converts an object to a torch::DataLoader
#'
#' @description
#' It takes a task and converts it first to a dataset and then to a dataloader
#' @export
#' @param x (`any`) object to be converted to a dataloader.
as_dataloader = function(x, row_ids, batch_size, device, ...) {
  UseMethod("as_dataloader")
}

#' @export
as_dataloader.Task = function(x, batch_size, device, row_ids, ...) { # nolint
  # TODO: Check that arguments go correctly into as_dataset and as_dataloader (argument names
  # must be disjunct --> what if not??? -> they must be listed explicitly like batch_size below)
  dataset = as_dataset(x, batch_size = batch_size, device = device, row_ids = row_ids)
  as_dataloader(dataset, batch_size = batch_size, ...)
}

as_dataloader.DataBackend = function(x, sets, shuffle, drop_last, target, features) { # nolint
  stop("Not implemented yet.")
}

#' @export
as_dataloader.DataBackendDataLoader = function(x, target, features, shuffle, drop_last, ...) { # nolint
  # this Backend has to be implemented to allow for arbitrary DataLoaders in Tasks
  stop("Not implemented yet.")
}




#' @export
as_dataloader.dataset = function(x, batch_size, ...) { # nolint
  dataloader(
    data = x,
    batch_size = batch_size,
    ...
  )
}
