#' @export
#' @include
# DataBackendTorchDataTable = R6Class(
#   inherit = DataBackendDataTable,
#   public = list(
#     initialize = function(data, primary_key) {
#       super$initialize(data, primary_key)
#     },
#     dataloader = function(rows, target, features, batch_size, device) {
#       rows = assert_integerish(rows, coerce = TRUE)
#       cols = c(target, features)
#       assert_names(cols, type = "unique")
#       cols = intersect(cols, colnames(private$.data))
#
#       .__i__ = keep_in_bounds(rows, 1L, nrow(private$.data))
#       # TODO:
#       # here features are reordered (if order in task$col_roles$feature differs from dataframe)
#       # might wanna change that ...
#       data = self$dataset(.__i__, target, features, batch_size, device)
#       sampler = SequentialSampler$new(data)
#       dataloader(
#         data = data,
#         sampler = sampler,
#         batch_size = batch_size
#       )
#     },
#     print = function() {
#       catf("<DataBackendTorchDataTable> (%sx%s)", self$nrow, self$ncol)
#       print(self$head(6L), row.names = FALSE, print.keys = FALSE)
#       # TODO: also print rows and cols
#       if (nrow(private$.data) > 6L) {
#         catf("[...] (%i rows omitted)", self$nrow - 6L)
#       }
#     },
#     dataset = function(rows, target, features, batch_size, device) {
#       # TODO: only create dataset for those rows that are used (i.e. the rows for the dataloader)
#       make_dataset(
#         rows = rows,
#         target = target,
#         data = private$.data,
#         features = features,
#         batch_size = batch_size,
#         device = device
#       )()
#     }
#   )
# )

# Copied from torch
SequentialSampler = R6::R6Class(
  "utils_sampler_sequential",
  lock_objects = FALSE,
  inherit = torch:::Sampler,
  public = list(
    initialize = function(data_source) {
      self$data_source = data_source
    },
    .iter = function() {
      i = 0
      n = length(self$data_source)
      coro::as_iterator(seq_len(n))
    },
    .length = function() {
      length(self$data_source)
    }
  )
)

#' @export
# SequentialSubsetSampler = R6Class(
#   "utils_sampler_sequential",
#   lock_objects = FALSE,
#   inherit = torch:::Sampler,
#   public = list(
#     row_ids = NULL,
#     initialize = function(data_source, row_ids) {
#       self$data_source = data_source
#       self$row_ids = assert_integer(row_ids)
#     },
#     .iter = function() {
#       i = 0
#       coro::as_iterator(seq_len(length(self$data_source)))
#     },
#     .length = function() {
#       length(self$data_source)
#     }
#   )
# )

#' Creates a data_set from a data_table
make_dataset = function(data, target, features, batch_size, device, rows = NULL) {
  # data = data[, cols]
  # target_col = which(colnames(data) == target)
  # feature_cols = setdiff(seq_len(nrow(data)), target_col)
  is_numeric = map_lgl(data[, ..features], is.numeric)
  features_num = features[is_numeric]
  features_cat = features[!is_numeric]
  cols = c(target, features_num, features_cat)
  if (!is.null(rows)) {
    data = data[rows, ..cols]
  } else {
    data = data[, ..cols]
  }

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
      data = as.matrix(data[, ..target]),
      dtype = torch_float(),
      device = device
    )
  } else { # classification
    y = cat2tensor(data[, ..target], device = device)
  }

  data_list = list(
    y = y
  )

  if (!is.null(x_num)) {
    data_list[["x_num"]] = x_num
  }
  if (!is.null(x_cat)) {
    data_list[["x_cat"]] = x_cat
  }

  data_set = dataset(
    initialize = function() {
      self$data = data_list
    },
    .getitem = function(index) {
      map(self$data, function(tensor) tensor[index])
    },
    .length = function() {
      nrow(self$data[["y"]])
    }
  )
  return(data_set)
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
      col = as.factor(col)
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


# Creates a dataset from a data.table and then creates a dataloader from the dataset
make_dataloader = function(task, batch_size, device) {
  data = task$data() # already respects row_roles$use
  target = task$col_roles$target
  features = task$col_roles$feature
  data_set = make_dataset(data, target, features, batch_size, device)()
  sampler = SequentialSampler$new(data_set)
  data_loader = dataloader(
    data = data_set,
    sampler = sampler,
    batch_size = batch_size
  )
  return(data_loader)
}
