#' @title Backend for Torch Datasets
#' @description
#' Backend for Torch Datasets.
#' @export
DataBackendDataset = R6Class("DataBackendDataset",
  inherit = DataBackend,
  cloneable = FALSE,
  public = list(
    #' @descriotion
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
    initialize = function(data, colnames, primary_key = NULL, indices = seq_len(length(data))) {
      # FIXME: How to we ensure that the forward function knows what to do?
      assert_class(data, "dataset")
      private$.indices = assert_integerish(indices, len = length(data), unique = TRUE)

      if (!is.null(primary_key)) {
        assert_string(primary_key)
        assert_false(primary_key %in% colnames)
      }
      primary_key = unique_id("..row_id", colnames)
      private$.colnames = assert_character(colnames, any.missing = FALSE, min.len = 2)

      super$initialize(
        data = dataset,
        primary_key = primary_key,
        data_formats = "dataset"
      )
    },
    data = function(rows, cols = NULL, data_format = "dataset") {
      if (!is.null(cols) && !test_permutation(cols, private$.colnames)) {
        stopf("Subsetting of columns is not supported for objects of class DataBackendDataset, use a different backend.") # nolint
      }
      dataset_subset(private$.dataset, rows)
    },
    head = function(n = 6L) {
      dataset_subset(self$data, seq_len(n))
    },
    missings = function(rows, cols) {
      set_names(0, cols)
    },
    distinct = function(rows, cols, na_rm = TRUE) {
      stopf("DataBackendDataset does not support the `$distinct()` method.")
    },
    ncol = function() length(private$.colnames),
    nrow = function() length(private$.dataset),
    rownames = function() {
      seq_len(self$nrow)
    },
    colnames = function() {
      c(self$.colnames, private$primary_key)
    }
  ),
  private = list(
    .colnames = NULL,
    .calculate_hash = function() {
      private$.data
    },
    .indices = NULL
  )
)

#' @export
col_info.DataBackendDataset = function(x, ...) { # nolint

}
