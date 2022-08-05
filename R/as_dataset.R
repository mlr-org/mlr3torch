#' @title Converts an object to a torch::Dataset
#' @param x (any)\cr Object to be converted to a [torch::dataset].
#' @param ... Additional arguments.
#' @export
as_dataset = function(x, ...) {
  UseMethod("as_dataset")
}

#' @export
as_dataset.Task = function(x, device = NULL, augmentation = NULL, row_ids = NULL, ...) { # nolint
  task = x
  feature_types = task$feature_types$type
  features = task$feature_names
  target = task$target_names

  data = task$data(row = row_ids) # already respects row_roles$use

  # currently ONLY images or ONLY tabular is supported
  if ("imageuri" %in% feature_types) {
    assert_true(length(task$feature_names) == 1L)
    ds = make_image_dataset(task, augmentation)
  } else if (all(feature_types %in% c("factor", "numeric", "integer", "logical", "ordered"))) {
    if (!is.null(augmentation)) {
      stopf("Augmentation is currently not supported for tabular data.")
    }
    ds = make_tabular_dataset(data, target, features, device)
  } else {
    stopf("Feature types currently not supported")
  }
  return(ds)
}
