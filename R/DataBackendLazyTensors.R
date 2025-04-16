
#' @title Data Backend for Lazy Tensors
#' @description
#' Special **experimental** data backend that converts [`lazy_tensor`] columns to their R representation.
#' However, [`LearnerTorch`] can directly operate on the lazy tensors.
#' @export
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

      self$data(n, self$colnames)
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
      for (nm in intersect(names(private$.converter), names(tbl))) {
        converted = private$.converter[[nm]](materialize(tbl[[nm]], rbind = TRUE))
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
col_info.DataBackendLazyTensors = function(x, ...) { # nolint
  first_row = x$head(1L)
  types = map_chr(first_row, function(x) class(x)[1L])
  discrete = setdiff(names(types)[types %chin% c("factor", "ordered")], x$primary_key)
  levels = insert_named(named_list(names(types)), map(first_row[discrete], levels))
  data.table(id = names(types), type = unname(types), levels = levels, key = "id")
}

#' @export
as_data_backend.dataset = function(x, dataset_shapes, primary_key ...) {


}