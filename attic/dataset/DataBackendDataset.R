#' @title Backend for Torch Datasets
#' @description
#' Backend for Torch Datasets.
DataBackendDataset = R6Class("DataBackendDataset",
  inherit = DataBackend,
  cloneable = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @param data (`torch::dataset`)\cr
    #'   The torch dataset.
    #' @param colnames (`character()`)\cr
    #'   The information returned by the data.loader.
    #'   FIXME: How to handle something like .index in task_dataset()? Should we allow colnames to be a subset?
    #'
    #' @param primary_key (`character(1)`)\cr
    #'   The name of the primary key, which is the vector provided by
    #'   Not really used, because we assume that
    #' @param indices (any)\cr
    #'   Unique identifiers for the data, i.e. those accepted by the `.getitem()` or `.getbatch()` method of
    #'   the provided dataset.
    initialize = function(data, colnames, primary_key = NULL, indices = seq_len(length(data)),
      col_info = NULL) {
      # FIXME: How to we ensure that the forward function knows what to do?
      assert_class(data, "dataset")
      private$.indices = assert_integerish(indices, len = length(data), unique = TRUE)

      if (is.null(col_info)) {
        batch = if (!is.null(data$.getbatch)) data$.getbatch(1) else data$.getitem(1)
        private$.col_info = data.table(
          id = names(batch),
          type = map_chr(batch, function(x) class(x)[[1L]]),
          dtype = map_chr(batch, function(x) as.character(x$dtype)),
          shape = map(batch, "shape"),
          levels = map(batch, function(x) if (x$dtype == torch_long()) unique(x)),
          fix_factor_levels = FALSE,
          labels = NA_character_,
          key = "id"
        )
      } else {
        # FIXME: assertions
        private$.col_info = assert_data_table(col_info, cols = c("id", "dtype", "shape"))
      }

      if (!is.null(primary_key)) {
        assert_string(primary_key)
        assert_false(primary_key %in% colnames)
      } else {
        primary_key = unique_id("..row_id", colnames)

      }
      private$.colnames_ds = assert_character(colnames, any.missing = FALSE, min.len = 2)

      super$initialize(
        data = data,
        primary_key = primary_key,
        data_formats = "dataset"
      )
    },
    #' @param features (`NULL` or `character()`)\cr
    #'   The features.
    #' Either the target and features are provided or none.
    #' @param target (`NULL` or `character(1)`)\cr
    #'   The target.
    #'   Either the target and features are provided or none.
    data = function(rows, cols = NULL, data_format = "dataset", device = "cpu", structure = NULL) {
      rows = assert_integerish(rows, coerce = TRUE)
      assert_names(cols, type = "unique")
      assert_choice(data_format, self$data_formats)
      cols = intersect(cols, self$colnames)

      datasubset(
        dataset = private$.data,
        rows = rows,
        cols = cols,
        device = device,
        structure = NULL
      )
    },
    head = function(n = 6L) {
      datasubset(private$.data, seq_len(n), self$colnames)
    },
    missings = function(rows, cols) {
      set_names(0, cols)
    },
    distinct = function(rows, cols, na_rm = TRUE) {
      stopf("DataBackendDataset does not support the `$distinct()` method.")
    }
    # TODO: printer
  ),
  active = list(
    ncol = function() length(private$.colnames),
    nrow = function() length(private$.data),
    rownames = function() {
      seq_len(self$nrow)
    },
    colnames = function() {
      c(private$.colnames_ds, private$primary_key)
    }
  ),
  private = list(
    .colnames_ds = NULL,
    .indices = NULL,
    .col_info = NULL,
    .calculate_hash = function() {
      private$.data
    }
  )
)

#' @export
col_info.DataBackendDataset = function(x, ...) { # nolint
  get_private(x)$.col_info
}
