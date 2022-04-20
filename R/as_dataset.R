#' @title Converts an object to a torch::Dataset
#' @export
as_dataset = function(x, ...) {
  UseMethod("as_dataset")
}

#' @export
as_dataset.Task = function(x, batch_size, device = NULL, row_ids = NULL, ...) { # nolint
  task = x
  feature_types = task$feature_types$type
  features = task$feature_names
  target = task$target_names

  data = task$data(row = row_ids) # already respects row_roles$use

  if ("imageuri" %in% feature_types) {
    stopf("Not supported yet.")
  } else if (all(feature_types %in% c("factor", "numeric", "integer", "logical"))) {
    dataset = make_tabular_dataset(data, target, features, batch_size, device)
  } else {
    stopf("Feature types not supported")
  }
  return(dataset)
}


#' @export
as_dataset.DataBackend = function(x, target, features, batch_size, device, ...) { # nolint
  stop("Not implemented yet.")
}
