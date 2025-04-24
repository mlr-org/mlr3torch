
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
    chunk_size = NULL,
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
    initialize = function(data, primary_key, converter, cache = names(converter), chunk_size = 100) {
      private$.converter = assert_list(converter, types = "function", any.missing = FALSE)
      assert_subset(names(converter), colnames(data))
      assert_subset(cache, names(converter), empty.ok = TRUE)
      private$.cached_cols = assert_subset(cache, names(converter))
      self$chunk_size = assert_int(chunk_size, lower = 1L)
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
      if (all(intersect(cols, private$.cached_cols) %in% names(private$.data_cache))) {
        expensive_cols = intersect(cols, private$.cached_cols)
        other_cols = setdiff(cols, expensive_cols)
        cache_hit = private$.data_cache[list(rows), expensive_cols, on = self$primary_key, with = FALSE]
        complete = complete.cases(cache_hit)
        cache_hit = cache_hit[complete]
        if (nrow(cache_hit) == length(rows)) {
          tbl = cbind(cache_hit, super$data(rows, other_cols))
          setcolorder(tbl, cols)
          return(tbl)
        }
        combined = rbindlist(list(cache_hit, private$.load_and_cache(rows[!complete], expensive_cols)))
        reorder = vector("integer", nrow(combined))
        reorder[complete] = seq_len(nrow(cache_hit))
        reorder[!complete] = nrow(cache_hit) + seq_len(nrow(combined) - nrow(cache_hit))

        tbl = cbind(combined[reorder], super$data(rows, other_cols))
        setcolorder(tbl, cols)
        return(tbl)
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
  active = list(
    converter = function(rhs) {
      assert_ro_binding(rhs)
      private$.converter
    }
  ),
  private = list(
    # call this function only with rows that are not in the cache yet
    .load_and_cache = function(rows, cols) {
      # Process columns that need conversion
      tbl = super$data(rows, cols)
      cols_to_convert = intersect(names(private$.converter), names(tbl))
      tbl_to_mat = tbl[, cols_to_convert, with = FALSE]
      # chunk the rows of tbl_to_mat into chunks of size self$chunk_size, apply materialize
      n = nrow(tbl_to_mat)
      chunks = split(seq_len(n), rep(seq_len(ceiling(n / self$chunk_size)), each = self$chunk_size, length.out = n))

      tbl_mat = if (n == 0) {
        set_names(list(torch_empty(0)), names(tbl_to_mat))
      } else {
        set_names(lapply(transpose_list(lapply(chunks, function(chunk) {
          materialize(tbl_to_mat[chunk, ], rbind = TRUE)
        })), torch_cat, dim = 1L), names(tbl_to_mat))
      }

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
as_task_classif.dataset = function(x, target, levels, converter = NULL, dataset_shapes = NULL, chunk_size = 100, cache = names(converter), ...) {
  if (length(x) < 2) {
    stopf("Dataset must have at least 2 rows.")
  }
  batch = dataloader(x, batch_size = 2)$.iter()$.next()
  if (is.null(converter)) {
    if (length(levels) == 2) {
      if (batch[[target]]$dtype != torch_float()) {
        stopf("Target must be a float tensor, but has dtype %s", batch[[target]]$dtype)
      }
      if (test_equal(batch[[target]]$shape, c(2L, 1L))) {
        converter = set_names(list(crate(function(x) factor(as.integer(x), levels = 0:1, labels = levels), levels)), target)
      } else {
        stopf("Target must be a float tensor of shape (batch_size, 1), but has shape (batch_size, %s)",
          paste(batch[[target]]$shape[-1L], collapse = ", "))
      }
      converter = set_names(list(crate(function(x) factor(as.integer(x), levels = 0:1, labels = levels), levels)), target)
    } else {
      if (batch[[target]]$dtype != torch_int()) {
        stopf("Target must be an integer tensor, but has dtype %s", batch[[target]]$dtype)
      }
      if (test_equal(batch[[target]]$shape, 2L)) {
        converter = set_names(list(crate(function(x) factor(as.integer(x), labels = levels), levels)), target)
      } else {
        stopf("Target must be an integer tensor of shape (batch_size), but has shape (batch_size, %s)",
          paste(batch[[target]]$shape[-1L], collapse = ", "))
      }
      converter = set_names(list(crate(function(x) factor(as.integer(x), labels = levels), levels)), target)
    }
  }
  be = as_data_backend(x, dataset_shapes, converter = converter, cache = cache, chunk_size = chunk_size)
  as_task_classif(be, target = target, ...)
}

#' @export
as_task_regr.dataset = function(x, target, converter = NULL, dataset_shapes = NULL, chunk_size = 100, cache = names(converter), ...) {
  if (length(x) < 2) {
    stopf("Dataset must have at least 2 rows.")
  }
  if (is.null(converter)) {
    converter = set_names(list(as.numeric), target)
  }
  batch = dataloader(x, batch_size = 2)$.iter()$.next()

  if (batch[[target]]$dtype != torch_float()) {
    stopf("Target must be a float tensor, but has dtype %s", batch[[target]]$dtype)
  }

  if (!test_equal(batch[[target]]$shape, c(2L, 1L))) {
    stopf("Target must be a float tensor of shape (batch_size, 1), but has shape (batch_size, %s)",
      paste(batch[[target]]$shape[-1L], collapse = ", "))
  }

  dataset_shapes = get_or_check_dataset_shapes(x, dataset_shapes)
  be = as_data_backend(x, dataset_shapes, converter = converter, cache = cache, chunk_size = chunk_size)
  as_task_regr(be, target = target, ...)
}

#' @export
col_info.DataBackendLazyTensors = function(x, ...) { # nolint
  first_row = x$head(1L)
  types = map_chr(first_row, function(x) class(x)[1L])
  discrete = setdiff(names(types)[types %chin% c("factor", "ordered")], x$primary_key)
  levels = insert_named(named_list(names(types)), map(first_row[, discrete, with = FALSE], levels))
  data.table(id = names(types), type = unname(types), levels = levels, key = "id")
}


# conservative check that avoids that a pseudo-lazy-tensor is preprocessed by some pipeop
# @param be
#   the backend
# @param candidates
#   the feature and target names
# @param visited
#  Union of all colnames already visited
# @return visited
check_lazy_tensors_backend = function(be, candidates, visited = character()) {
  if (inherits(be, "DataBackendRbind") || inherits(be, "DataBackendCbind")) {
    bs = be$.__enclos_env__$private$.data
    # first we check b2, then b1, because b2 possibly overshadows some b1 rows/cols
    visited = check_lazy_tensors_backend(bs$b2, candidates, visited)
    check_lazy_tensors_backend(bs$b1, candidates, visited)
  } else {
    if (inherits(be, "DataBackendLazyTensors")) {
      if (any(names(be$converter) %in% visited)) {
        converter_cols = names(be$converter)[names(be$converter) %in% visited]
        stopf("A converter column ('%s') from a DataBackendLazyTensors was presumably preprocessed by some PipeOp. This can cause inefficiencies and is therefore not allowed. If you want to preprocess them, please directly encode them as R types.", paste0(converter_cols, collapse = ", ")) # nolint
      }
    }
    union(visited, intersect(candidates, be$colnames))
  }
}
