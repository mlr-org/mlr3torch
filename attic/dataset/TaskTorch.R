TaskClassifTorch = R6Class("TaskClassifTorch",
  inherit = TaskSupervised,
  public = list(
    initialize = function(backend, id, target) {
      super$initialize(
        id = id,
        task_type = "classif",
        backend = backend,
        target = target
      )
    },
    # FIXME: Test that Task$data arguments stay the same
    data = function(rows = NULL, cols = NULL, data_format = "dataset", ordered = FALSE, device) {
      mlr3:::assert_has_backend(self)
      assert_choice(data_format, self$data_formats)
      assert_flag(ordered)

      row_roles = self$row_roles
      col_roles = self$col_roles

      if (is.null(rows)) {
        rows = row_roles$use
      } else {
        assert_subset(rows, self$backend$rownames)
        if (is.double(rows)) {
          rows = as.integer(rows)
        }
      }

      if (is.null(cols)) {
        query_cols = cols = c(col_roles$target, col_roles$feature)
      } else {
        assert_subset(cols, self$col_info$id)
        query_cols = cols
      }

      reorder_rows = length(col_roles$order) > 0L && ordered
      if (reorder_rows) {
        if (data_format != "data.table") {
          stopf("Ordering only supported for data_format 'data.table'")
        }
        query_cols = union(query_cols, col_roles$order)
      }

      data = self$backend$data(
        rows = rows,
        cols = query_cols,
        data_format = data_format,
        device = device,
        structure = c(x = self$feature_names, y = self$target_names)
        )

      # CHANGE: change nrow(dataset) to length(datataset)
      if (length(query_cols) && data$nrow() != length(rows)) {
        stopf("DataBackend did not return the queried rows correctly: %i requested, %i received", length(rows), nrow(data))
      }

      if (length(rows) && data$ncol() != length(query_cols)) {
        stopf("DataBackend did not return the queried cols correctly: %i requested, %i received", length(cols), ncol(data))
      }

      .__i__ = self$col_info[["fix_factor_levels"]]
      if (any(.__i__)) {
        fix_factors = self$col_info[.__i__, c("id", "levels"), with = FALSE][list(names(data)), on = "id", nomatch = NULL]
        if (nrow(fix_factors)) {
          data = fix_factor_levels(data, levels = set_names(fix_factors$levels, fix_factors$id))
        }
      }

      if (reorder_rows) {
        setorderv(data, col_roles$order)[]
        data = remove_named(data, setdiff(col_roles$order, cols))
      }

      return(data)
    }
  )
)
