make_tabular_dataset = function(data, target, features, device) {
  # data = data[, cols]
  # target_col = which(colnames(data) == target)
  # feature_cols = setdiff(seq_len(nrow(data)), target_col)
  is_numeric = map_lgl(data[, ..features], is.numeric)
  features_num = features[is_numeric]
  features_cat = features[!is_numeric]
  cols = c(target, features_num, features_cat)
  data = data[, ..cols]

  x_num = NULL
  x_cat = NULL
  if (length(features_num)) {
    x_num = torch_tensor(
      data = as.matrix(data[, ..features_num]),
      dtype = torch_float(),
      device = device
    )
  }
  if (length(features_cat)) {
    x_cat = cat2tensor(data[, ..features_cat], device = device)
  }

  if (is.numeric(data[[target]])) {
    y = torch_tensor(
      data = data[, ..target][[1L]],
      dtype = torch_float(),
      device = device
    )
  } else { # classification
    y = cat2tensor(data[, ..target], device = device)[,1]
  }

  data_list = list(y = y)

  if (!is.null(x_num)) {
    data_list[["num"]] = x_num
  }
  if (!is.null(x_cat)) {
    data_list[["cat"]] = x_cat
  }

  dataset(
    initialize = function() {
      self$data = data_list
      self$feature_types = setdiff(names(data_list), "y")
    },
    .getbatch = function(index) {
      list(
        y = self$data$y[index, drop = FALSE],
        x = Map(function(type) self$data[[type]][index, drop = FALSE], self$feature_types)
      )
    },
    .length = function() {
      nrow(self$data[["y"]])
    }
  )()
}

cat2tensor = function(data, device) {
  classes = map_chr(data, function(x) class(x)[[1]])
  # TODO: how to deal with integers?
  assert_true(all(classes %nin% c("numeric", "integer")))
  assert(ncol(data) > 0)
  change_cols = seq_len(ncol(data))
  data = copy(data)
  encode = function(col) {
    if (is.character(col)) {
      stop("Not implemented yet")
      # here we have to be careful what happens if certain characters don't appear in the
      # train set e.g., it needs to be ensured that the characters are always encoded correctly
      # col = as.factor(col)
    }
    col = as.integer(col)
    return(col)
  }

  if (length(change_cols)) {

  }
  data[, (change_cols) := lapply(.SD, encode), .SDcols = change_cols]
  tensor = torch_tensor(
    data = as.matrix(data),
    dtype = torch_long(),
    device = device
  )
  return(tensor)
}
