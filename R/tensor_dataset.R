tensor_dataset = dataset("tensor_datset",
  initialize = function(dataset) {
    assert_class(dataset, "dataset")
    # the return of dataset is list(x = list<torch_tensor>, y = torch_float, .index = torch_long)
    self$data = if (!is.null(dataset$.getbatch)) {
      dataset$.getbatch(seq_len(length(dataset)))
    } else {
      batches = lapply(seq_len(length(dataset)), dataset$.getitem)
      list(
        x = map(batches$x, torch_stack),
        y  = torch_stack(batches$y),
        .index = torch_stack(batches$.index)
      )
    }
  },
  .getbatch = function(i) {
    list(
      x = map(self$data$x, function(x) x[i, drop = FALSE]),
      y = self$data$y[i, drop = FALSE],
      .index = self$data$index[i, drop = FALSE]
    )
  },
  .length = function() {
    nrow(self$data$x[[1L]])
  }
)

