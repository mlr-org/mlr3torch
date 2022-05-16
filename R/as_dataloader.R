#' @title Converts an object to a torch::DataLoader
#'
#' @description
#' It takes a task and converts it first to a dataset and then to a dataloader
#' @export
#' @param x (`any`) object to be converted to a dataloader.
as_dataloader = function(x, ...) {
  UseMethod("as_dataloader")
}

#' @export
as_dataloader.Task = function(x, batch_size, device, row_ids = NULL, ...) { # nolint
  # TODO: Check that arguments go correctly into as_dataset and as_dataloader (argument names
  # must be disjunct --> what if not??? -> they must be listed explicitly like batch_size below)
  dataset = as_dataset(x, batch_size = batch_size, device = device, row_ids = row_ids)
  as_dataloader(dataset, batch_size = batch_size, ...)
}

#' @export
as_dataloader.dataset = function(x, batch_size, device, ...) { # nolint
  dataloader(
    dataset = x,
    batch_size = batch_size,
    ...
  )
}
