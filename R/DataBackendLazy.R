#' @title Lazy Data Backend
#'
#' @name mlr_backends_lazy
#'
#' @description
#' This lazy data backend wraps a constructor that lazily creates another backend, e.g. by downloading
#' (and caching) some data from the internet.
#' This backend should be used, when some metadata of the backend is known in advance and should be accessible
#' before downloading the actual data.
#' When the backend is first constructed, it is verified that the provided metadata was correct, otherwise
#' an informative error message is thrown.
#' After the construction of the lazily constructed backend, calls like `$data()`, `$missings()`, `$distinct()`,
#' or `$hash()` are redirected to it.
#'
#' Information that is available before the backend is constructed is:
#' * `nrow` - The number of rows (set as the length of the `rownames`).
#' * `ncol` - The number of columns (provided via the `id` column of `col_info`).
#' * `colnames` - The column names.
#' * `rownames` - The row names.
#' * `col_info` - The column information, which can be obtained via [`mlr3::col_info()`].
#'
#' Beware that accessing the backend's hash also contructs the backend.
#'
#' Note that while in most cases the data contains [`lazy_tensor`] columns, this is not necessary and the naming
#' of this class has nothing to do with the [`lazy_tensor`] data type.
#'
#' **Important**
#'
#' When the constructor generates `factor()` variables it is important that the ordering of the levels in data
#' corresponds to the ordering of the levels in the `col_info` argument.
#'
#' @param constructor (`function`)\cr
#'   A function with argument `backend` (the lazy backend), whose return value must be the actual backend.
#'   This function is called the first time the field `$backend` is accessed.
#' @param rownames (`integer()`)\cr
#'   The row names. Must be a permutation of the rownames of the lazily constructed backend.
#' @param col_info ([`data.table::data.table()`])\cr
#'   A data.table with columns `id`, `type` and `levels` containing the column id, type and levels.
#'   Note that the levels must be provided in the correct order.
#' @param cols (`character()`)\cr
#'   Column names.
#' @param rows (`integer()`)\cr
#'   Row indices.
#' @param data_format (`character(1)`)\cr
#'  Desired data format, e.g. `"data.table"` or `"Matrix"`.
#' @param na_rm (`logical(1)`)\cr
#'   Whether to remove NAs or not.
#' @param data_formats (`character()`)\cr
#'   Set of supported data formats. E.g. `"data.table"`.
#'   These must be a subset of the data formats of the lazily constructed backend.
#' @param primary_key (`character(1)`)\cr
#'   Name of the primary key column.
#'
#' @export
#' @examplesIf torch::torch_is_installed()
#' # We first define a backend constructor
#' constructor = function(backend) {
#'   cat("Data is constructed!\n")
#'   DataBackendDataTable$new(
#'     data.table(x = rnorm(10), y = rnorm(10), row_id = 1:10),
#'     primary_key = "row_id"
#'   )
#' }
#'
#' # to wrap this backend constructor in a lazy backend, we need to provide the correct metadata for it
#' column_info = data.table(
#'   id = c("x", "y", "row_id"),
#'   type = c("numeric", "numeric", "integer"),
#'   levels = list(NULL, NULL, NULL)
#' )
#' backend_lazy = DataBackendLazy$new(
#'   constructor = constructor,
#'   rownames = 1:10,
#'   col_info = column_info,
#'   data_formats = "data.table",
#'   primary_key = "row_id"
#' )
#'
#' # Note that the constructor is not called for the calls below
#' # as they can be read from the metadata
#' backend_lazy$nrow
#' backend_lazy$rownames
#' backend_lazy$ncol
#' backend_lazy$colnames
#' col_info(backend_lazy)
#'
#' # Only now the backend is constructed
#' backend_lazy$data(1, "x")
#' # Is the same as:
#' backend_lazy$backend$data(1, "x")
DataBackendLazy = R6Class("DataBackendLazy",
  inherit = DataBackend,
  cloneable = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(constructor, rownames, col_info, primary_key, data_formats) {
      private$.rownames = assert_integerish(rownames, unique = TRUE)
      private$.col_info = assert_data_table(col_info, ncols = 3, min.rows = 1)
      assert_permutation(colnames(col_info), c("id", "type", "levels"))
      assert_choice(primary_key, col_info$id)
      private$.colnames = col_info$id
      assert_choice(primary_key, col_info$id)
      private$.constructor = assert_function(constructor, args = "backend")

      super$initialize(data = NULL, primary_key = primary_key, data_formats = data_formats)
    },

    #' @description
    #' Returns a slice of the data in the specified format.
    #' The rows must be addressed as vector of primary key values, columns must be referred to via column names.
    #' Queries for rows with no matching row id and queries for columns with no matching column name are silently ignored.
    #' Rows are guaranteed to be returned in the same order as `rows`, columns may be returned in an arbitrary order.
    #' Duplicated row ids result in duplicated rows, duplicated column names lead to an exception.
    #'
    #' Accessing the data triggers the construction of the backend.
    data = function(rows, cols, data_format = "data.table") {
      self$backend$data(rows = rows, cols = cols, data_format = data_format)
    },

    #' @description
    #' Retrieve the first `n` rows.
    #' This triggers the construction of the backend.
    #'
    #' @param n (`integer(1)`)\cr
    #'   Number of rows.
    #'
    #' @return [data.table::data.table()] of the first `n` rows.
    head = function(n = 6L) {
      self$backend$head(n = n)
    },
    #' @description
    #' Returns a named list of vectors of distinct values for each column
    #' specified. If `na_rm` is `TRUE`, missing values are removed from the
    #' returned vectors of distinct values. Non-existing rows and columns are
    #' silently ignored.
    #'
    #' This triggers the construction of the backend.
    #'
    #' @return Named `list()` of distinct values.
    distinct = function(rows, cols, na_rm = TRUE) {
      self$backend$distinct(rows = rows, cols = cols, na_rm = na_rm)
    },
    #' @description
    #' Returns the number of missing values per column in the specified slice
    #' of data. Non-existing rows and columns are silently ignored.
    #'
    #' This triggers the construction of the backend.
    #'
    #' @return Total of missing values per column (named `numeric()`).
    missings = function(rows, cols) {
      self$backend$missings(rows = rows, cols = cols)
    },
    #' @description
    #' Printer.
    print = function() {
      nr = self$nrow
      catf("%s (%ix%i)", format(self), nr, self$ncol)
      if (is.null(private$.backend)) {
        catf(" * Backend not loaded yet.")
      } else {
        catf(" * Underlying backend: <%s>", class(self$backend)[[1L]])
        print(self$head(6L), row.names = FALSE, print.keys = FALSE)
        if (nr > 6L) {
          catf("[...] (%i rows omitted)", nr - 6L)
        }
      }
    }
  ),
  active = list(
    #' @field backend (`DataBackend`)\cr
    #'   The wrapped backend that is lazily constructed when first accessed.
    backend = function(rhs) {
      assert_ro_binding(rhs)
      if (is.null(private$.backend)) {
        backend = assert_backend(private$.constructor(self))

        f = function(test, x, y, var_name) {
          if (!test(x, y)) {
            stopf(paste0(
              "The '%s' was/were specified incorrecly during construction.\n",
              "Observed for constructed backend:\n'%s'\n",
              "Specified during construction:\n'%s'"
            ), var_name, paste0(capture.output(x), collapse = "\n"), paste0(capture.output(y), collapse = "\n"))
          }
        }

        f(identical, backend$primary_key, self$primary_key, "primary key")
        f(test_permutation, backend$rownames, self$rownames, "row identifiers")
        f(test_permutation, backend$colnames, private$.colnames, "column names")
        f(test_equal_col_info, col_info(backend), private$.col_info, "column information")
        # need to reverse the order for correct error message
        f(function(x, y) test_subset(y, x), backend$data_formats, self$data_formats, "data formats")
        private$.backend = backend
      }
      private$.backend
    },
    #' @field nrow (`integer(1)`)\cr
    #' Number of rows (observations).
    nrow = function(rhs) {
      assert_ro_binding(rhs)
      length(private$.rownames)
    },
    #' @field ncol (`integer(1)`)\cr
    #' Number of columns (variables), including the primary key column.
    ncol = function(rhs) {
      assert_ro_binding(rhs)
      length(private$.colnames)
    },
    #' @field rownames (`integer()`)\cr
    #' Returns vector of all distinct row identifiers, i.e. the contents of the primary key column.
    rownames = function(rhs) {
      assert_ro_binding(rhs)
      private$.rownames
    },
    #' @field colnames (`character()`)\cr
    #' Returns vector of all column names, including the primary key column.
    colnames = function(rhs) {
      assert_ro_binding(rhs)
      private$.colnames
    },
    #' @field is_constructed (`logical(1)`)\cr
    #'   Whether the backend has already been constructed.
    is_constructed = function(rhs) {
      assert_ro_binding(rhs)
      !is.null(private$.backend)
    }
  ),
  private = list(
    .calculate_hash = function() {
      get_private(self$backend)$.calculate_hash()
    },
    .constructor = NULL,
    .backend = NULL,
    .rownames = NULL,
    .colnames = NULL,
    .col_info = NULL
  )
)

#' @export
col_info.DataBackendLazy = function(x, ...) {
  copy(get_private(x)$.col_info)
}
