
#' @title Special Backend for Lazy Tensors
#' @description
#' This backend essentially allows you to use a [`torch::dataset`] directly with
#' an [`mlr3::Learner`].
#'
#' * The data cannot contain missing values, as [`lazy_tensor`]s do not support them.
#'   For this reason, calling `$missings()` will always return `0` for all columns.
#' * The `$distinct()` method will consider two lazy tensors that refer to the same element of a
#'   [`DataDescriptor`] to be identical.
#'   This means, that it might be underreporting the number of distinct values of lazy tensor columns.
#'
#' @export
#' @examplesIf torch::torch_is_installed()
#' # used as feature in all backends
#' x = torch_randn(100, 10)
#' # regression
#' ds_regr = tensor_dataset(x = x, y = torch_randn(100, 1))
#' be_regr = as_data_backend(ds_regr, converter = list(y = as.numeric))
#' be_regr$head()
#'
#'
#' # binary classification: underlying target tensor must be float in [0, 1]
#' ds_binary = tensor_dataset(x = x, y = torch_randint(0, 2, c(100, 1))$float())
#' be_binary = as_data_backend(ds_binary, converter = list(
#'   y = function(x) factor(as.integer(x), levels = c(0, 1), labels = c("A", "yes"))
#' ))
#' be_binary$head()
#'
#' # multi-class classification: underlying target tensor must be integer in [1, K]
#' ds_multiclass = tensor_dataset(x = x, y = torch_randint(1, 4, size = c(100, 1)))
#' be_multiclass = as_data_backend(ds_multiclass, converter = list(y = as.numeric))
#' be_multiclass$head()

DataBackendLazyTensors = R6Class("DataBackendLazyTensors",
  cloneable = FALSE,
  inherit = DataBackendDataTable,
  public = list(
    #' @description
    #' Create a new instance of this [R6][R6::R6Class] class.
    #' @param data (`data.table`)\cr
    #'   Data containing (among others) [`lazy_tensor`] columns.
    #' @param primary_key (`character(1)`)\cr
    #'   Name of the column used as primary key.
    #' @param converter (named `list()` of `function`s)\cr
    #'   A named list of functions that convert the lazy tensor columns to their R representation.
    #'   The names must be the names of the columns that need conversion.
    #' @param cache (`character()`)\cr
    #'   Names of the columns that should be cached.
    #'   Per default, all columns that are converted are cached.
    initialize = function(data, primary_key, converter, cache = names(converter)) {
      private$.converter = assert_list(converter, types = "function", any.missing = FALSE)
      assert_subset(names(converter), colnames(data))
      private$.cached_cols = assert_subset(cache, names(converter))
      walk(names(private$.converter), function(nm) {
        if (!inherits(data[[nm]], "lazy_tensor")) {
          stopf("Column '%s' is not a lazy tensor.", nm)
        }
      })
      super$initialize(data, primary_key)
      # select the column whose name is stored in primary_key from private$.data but keep its name
      private$.data_cache = private$.data[, primary_key, with = FALSE]
    },
    data = function(rows, cols) {
      rows = assert_integerish(rows, coerce = TRUE)
      assert_names(cols, type = "unique")

      if (getOption("mlr3torch.data_loading", FALSE)) {
        # no caching, no materialization as this is called in the training loop
        return(super$data(rows, cols))
      }
      if (all(cols %in% names(private$.data_cache))) {
        cache_hit = private$.data_cache[list(rows), cols, on = self$primary_key, with = FALSE]
        complete = complete.cases(cache_hit)
        cache_hit = cache_hit[complete]
        if (nrow(cache_hit) == length(rows)) {
          return(cache_hit)
        }
        combined = rbindlist(list(cache_hit, private$.load_and_cache(rows[!complete], cols)))
        reorder = vector("integer", nrow(combined))
        reorder[complete] = seq_len(nrow(cache_hit))
        reorder[!complete] = nrow(cache_hit) + seq_len(nrow(combined) - nrow(cache_hit))
        return(combined[reorder])
      }

      private$.load_and_cache(rows, cols)
    },
    head = function(n = 6L) {
      if (getOption("mlr3torch.data_loading", FALSE)) {
        return(super$head(n))
      }

      self$data(seq_len(n), self$colnames)
    },
    missings = function(rows, cols) {
      set_names(rep(0L, length(cols)), cols)
    }
  ),
  private = list(
    # call this function only with rows that are not in the cache yet
    .load_and_cache = function(rows, cols) {
      # Process columns that need conversion
      tbl = super$data(rows, cols)
      cols_to_convert = intersect(names(private$.converter), names(tbl))
      tbl_to_mat = tbl[, cols_to_convert, with = FALSE]
      tbl_mat = materialize(tbl_to_mat, rbind = TRUE)

      for (nm in cols_to_convert) {
        converted = private$.converter[[nm]](tbl_mat[[nm]])
        tbl[[nm]] = converted

        if (nm %in% private$.cached_cols) {
          set(private$.data_cache, i = rows, j = nm, value = converted)
        }
      }
      return(tbl)
    },
    .data_cache = NULL,
    .converter = NULL,
    .cached_cols = NULL
  )
)

#' @export
as_data_backend.dataset = function(x, dataset_shapes, ...) {
  tbl = as_lazy_tensors(x, dataset_shapes, ...)
  tbl$row_id = seq_len(nrow(tbl))
  DataBackendLazyTensors$new(tbl, primary_key = "row_id", ...)
}

#' @export
as_task_classif.dataset = function(x, dataset_shapes, target, ...) {
  # TODO
}

#' @export
as_task_regr.dataset = function(x, dataset_shapes, target, converter, ...) {
  # TODO
}

#' @export
col_info.DataBackendLazyTensors = function(x, ...) { # nolint
  first_row = x$head(1L)
  types = map_chr(first_row, function(x) class(x)[1L])
  discrete = setdiff(names(types)[types %chin% c("factor", "ordered")], x$primary_key)
  levels = insert_named(named_list(names(types)), map(first_row[, discrete, with = FALSE], levels))
  data.table(id = names(types), type = unname(types), levels = levels, key = "id")
}