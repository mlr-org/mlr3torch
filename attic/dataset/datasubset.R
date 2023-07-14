# FIXME: This is adapted from torch I don't fully understand it
datasubset <- dataset("datasubset",
  initialize = function(dataset, rows, cols, device = "cpu", structure = NULL) {
    if (!is.null(structure)) {
      assert_list(structure, types = "character", names = "unique")
      assert_permutation(names(structure), c("x", "y"))
    }
    self$structure = structure
    self$x_names = structure$x
    self$y_names = structure$y

    self$rows = rows
    self$cols = cols
    self$dataset = dataset
    self$device = assert_choice(device, mlr_reflections$torch$devices)
    if (!is.null(dataset$.getbatch)) {
      self$.getbatch <- self$.getitem
    }
    classes = class(dataset)
    classes_to_append = classes[classes != "R6"]
    class(self) = c(paste0(classes_to_append, "_subset_toway"), class(self))
  },
  .getitem = function(idx) {
    # FIXME: disallow .index name for DataBackendDataset
    out = self$dataset[self$rows[idx]][self$cols]
    out = lapply(out, function(x) x$to(device = self$device))
    if (!is.null(self$structure)) {
      list(x = out[self$x_names], y = out[self$y_names], .index = idx)
    } else {
      c(out, list(.index = idx))
    }
  },
  .length = function() {
    return(length(self$rows))
  },
  ncol = function() {
    return(length(self$cols))
  },
  nrow = function() {
    return(length(self$rows))
  }
)

